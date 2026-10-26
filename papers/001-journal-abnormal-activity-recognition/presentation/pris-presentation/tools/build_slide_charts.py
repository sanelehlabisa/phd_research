#!/usr/bin/env python3
"""Two side-by-side slide figures only; never rebuild the user's deck or script.

Run with implementation/.venv/bin/python. Inputs are unchanged October 16 CSVs.
Search accuracy is the arithmetic mean over the four frame/FPS settings.
Final accuracy remains the saved test accuracy, not a newly evaluated metric.
"""
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from math import floor, isclose
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT.parents[1] / 'implementation/reports/2026-10-16'
FIGURES = PROJECT / 'assets'
AUDIT = PROJECT / 'tools'
BLACK = '#000000'
TEAL = '#007A68'
DEPTH_STYLE = {
    1: ('#0096B7', 'o'), 2: ('#2454C6', 's'), 3: (TEAL, '^'),
    4: ('#8D3DB0', 'D'), 5: ('#D06400', 'P'),
}
FINAL_STYLE = {
    'custom_top1': (TEAL, 'o'),
    'custom_top2': ('#7C2EAC', 'o'),
    'custom_top3': ('#1868CE', 'o'),
    'custom_top4': ('#388425', 'o'),
    'custom_top5': ('#AF7800', 'o'),
    'paper_convlstm_adapted': ('#555555', 'D'),
    'r3d_18': ('#974A18', 's'),
    'mc3_18': ('#E06400', 's'),
    'swin3d_t': ('#AE438E', '^'),
    'swin3d_s': ('#CE2656', '^'),
}


def read_csv(name):
    with (REPORT / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_rows():
    raw = read_csv('search_results.csv')
    grouped = defaultdict(list)
    for row in raw:
        assert row['size'] == '32' and row['seed'] == '42'
        grouped[row['architecture']].append(row)
    assert len(raw) == 128 and len(grouped) == 32
    rows = []
    for architecture, settings in grouped.items():
        assert len(settings) == 4
        assert {(r['frames'], r['fps']) for r in settings} == {
            ('8','8'), ('8','16'), ('16','8'), ('16','16')}
        assert len({r['parameters'] for r in settings}) == 1
        scores = [float(r['validation_accuracy']) for r in settings]
        rows.append({
            'model': settings[0]['model'], 'architecture': architecture,
            'depth': int(settings[0]['depth']), 'parameters': int(settings[0]['parameters']),
            'accuracy': mean(scores), 'minimum_accuracy': min(scores),
            'mean_validation_loss': mean(float(r['validation_loss']) for r in settings),
            'settings': [{'frames': int(r['frames']), 'fps': int(r['fps']),
                          'validation_accuracy': float(r['validation_accuracy'])} for r in settings],
        })
    rows.sort(key=lambda r: (-r['accuracy'], -r['minimum_accuracy'],
                            r['mean_validation_loss'], r['parameters'], r['model']))
    saved = sorted(read_csv('architecture_ranking.csv'), key=lambda r: int(r['rank']))
    for row, reference in zip(rows, saved):
        assert row['architecture'] == reference['architecture']
        assert isclose(row['accuracy'], float(reference['mean_validation_accuracy']), abs_tol=1e-12)
    assert rows[0]['architecture'] == '64-64-128' and isclose(rows[0]['accuracy'], .9)
    return rows


def final_rows():
    raw = read_csv('comparison_results.csv')
    assert len(raw) == 10 and {r['model'] for r in raw} == set(FINAL_STYLE)
    rows = []
    for r in raw:
        assert (r['frames'],r['fps'],r['size'],r['seed']) == ('8','16','64','42')
        rows.append({'model':r['model'], 'architecture':r['architecture'],
                     'parameters':int(r['parameters']), 'accuracy':float(r['test_accuracy'])})
    rows.sort(key=lambda r: (-r['accuracy'],r['parameters'],r['model']))
    assert rows[0]['model'] == 'custom_top1'
    return rows


def canvas(title, subtitle, ylabel):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,
                         'text.color':BLACK,'axes.labelcolor':BLACK,
                         'axes.titlecolor':BLACK,'legend.labelcolor':BLACK,
                         'xtick.color':BLACK,'ytick.color':BLACK,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig = plt.figure(figsize=(18,11),dpi=150,facecolor='white')
    fig.text(.355,.965,title,ha='center',va='top',fontsize=29,weight='bold')
    fig.text(.355,.908,subtitle,ha='center',va='top',fontsize=20,weight='normal')
    ax = fig.add_axes([.08,.200,.55,.650])
    ax.spines[['top','right']].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#B8C7D2')
    ax.set_xscale('log')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y',color='#E2E9EF',linewidth=1,zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=10,labelsize=18)
    ax.set_xlabel('Parameters (millions, log scale)',fontsize=21,labelpad=14)
    ax.set_ylabel(ylabel,fontsize=21,labelpad=16)
    return fig,ax


def layer_key(fig):
    handles = [Line2D([],[],color=c,marker=m,linestyle='none',markersize=13,
                       markeredgecolor='#202020',markeredgewidth=.65,
                       label=f'{d} layer'+('s' if d != 1 else ''))
               for d,(c,m) in DEPTH_STYLE.items()]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.355,.085),ncol=5,
               frameon=False,fontsize=17,columnspacing=1.3,handletextpad=.3)


def family_key(fig):
    families = [('Custom ConvLSTM','o'),('Adapted reference','D'),
                ('3D CNN','s'),('ViT','^')]
    handles = [Line2D([],[],color=BLACK,marker=m,linestyle='none',
                      markersize=13,label=label) for label,m in families]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.355,.085),ncol=4,
               frameon=False,fontsize=17,columnspacing=1.3,handletextpad=.3)


def point(ax,row,color,marker,winner=False):
    x,y=row['parameters']/1e6,row['accuracy']*100
    ax.scatter(x,y,s=225,color=color,marker=marker,edgecolors='#202020',linewidths=.65,zorder=3)
    if winner:
        ax.scatter(x,y,s=800,facecolors='none',edgecolors=TEAL,linewidths=2.8,zorder=4)


def model_name(row,search):
    if search or row['model'].startswith('custom_'):
        widths='['+row['architecture'].replace('-',',')+']'
        return widths if search else f'ConvLSTM{widths}'
    return {'paper_convlstm_adapted':'Adapted reference',
            'r3d_18':'R3D-18','mc3_18':'MC3-18',
            'swin3d_t':'Swin3D-T','swin3d_s':'Swin3D-S'}[row['model']]


def model_keys(fig,rows,search):
    """Ten ranked model cards in one vertical column to the plot's right."""
    fig.text(.685,.949,'Top 10 search models' if search else 'Comparison models',
             fontsize=22,weight='bold',va='center')
    for i,row in enumerate(rows[:10]):
        x=.685
        y=.850-i*.082
        width,height=.300,.064
        color,marker=DEPTH_STYLE[row['depth']] if search else FINAL_STYLE[row['model']]
        border=FancyBboxPatch((x,y),width,height,transform=fig.transFigure,
                             boxstyle='round,pad=0.0015,rounding_size=0.005',
                             facecolor='#F0F7F7' if i==0 else '#F7F9FB',
                             edgecolor=TEAL if i==0 else '#E2E9EF',
                             linewidth=2.2 if i==0 else .8,zorder=1)
        fig.add_artist(border)
        fig.add_artist(Line2D([x+.014],[y+height*.57],transform=fig.transFigure,
                              marker=marker,color=color,linestyle='none',
                              markersize=12,markeredgecolor='#202020',
                              markeredgewidth=.65,zorder=2))
        fig.text(x+.031,y+.044,f'{i+1:02d}  {model_name(row,search)}',
                 fontsize=18,weight='bold' if i==0 else 'normal',va='center')
        accuracy=(Decimal(str(row['accuracy']))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        fig.text(x+.031,y+.017,
                 f"{accuracy}%  ·  {row['parameters']/1e6:.2f}M parameters",
                 fontsize=17,color=BLACK,va='center')


def save(fig,name):
    # The two requested presentation charts are the only image outputs.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for artist in fig.texts:
        bounds=artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    fig.savefig(FIGURES/f'{name}.png',dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    AUDIT.mkdir(parents=True,exist_ok=True)
    originals=[PROJECT/'script.txt',PROJECT/'SENG-F-12-HLABISA-SANELE.pptx',
               REPORT/'search_results.csv',REPORT/'comparison_results.csv',REPORT/'architecture_ranking.csv']
    originals.extend(p for p in (PROJECT/'SENG-F-12-HLABISA-SANELE.pdf',
                                 FIGURES/'gradio_app_qr.png',FIGURES/'surveillance_operator.png',
                                 FIGURES/'cover_detection.png') if p.is_file())
    before={str(p):sha256(p) for p in originals}
    search,final=search_rows(),final_rows()
    fig,ax=canvas('Search', '[8, 16] frames · [8, 16] FPS · 32 × 32 px',
                  'Mean validation accuracy (%)')
    for i,row in enumerate(search):
        point(ax,row,*DEPTH_STYLE[row['depth']],winner=i==0)
    ax.set_xlim(.032,4.4)
    ax.set_ylim(floor(min(r['accuracy'] for r in search)*100/10)*10,95)
    ax.xaxis.set_major_locator(FixedLocator([.04,.1,.3,1,3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    layer_key(fig)
    model_keys(fig,search,True)
    save(fig,'search')
    fig,ax=canvas('Comparison', '8 frames · 16 FPS · 64 × 64 px','Test accuracy (%)')
    for i,row in enumerate(final):
        point(ax,row,*FINAL_STYLE[row['model']],winner=i==0)
    ax.set_xlim(.2,200)
    ax.set_ylim(91,100.5)
    ax.set_yticks([92,94,96,98,100])
    ax.xaxis.set_major_locator(FixedLocator([.3,1,3,10,30,100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    family_key(fig)
    model_keys(fig,final,False)
    save(fig,'comparison')
    assert before=={str(p):sha256(p) for p in originals}, 'A protected input was changed'
    audit={'date':'2026-10-26','image_size_px':[2700,1650],
           'search_aggregation':'Arithmetic mean validation accuracy over exactly four frame/FPS conditions per architecture; no averaging of F1/class metrics.',
           'search_rows':search,'final_metric':'Unchanged saved test_accuracy',
           'final_rows':final,'source_and_protected_file_sha256':before,
           'model_cards_per_chart':10,'card_layout':'10 rows × 1 column to the right of the graph',
           'slides_and_script_unchanged':True}
    (AUDIT/'portrait_chart_data.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Saved assets/search.png and assets/comparison.png (2700×1650).')
    print('32 search points; all10 final points; sorted top10 cards on the right. Protected files unchanged.')


if __name__=='__main__':
    main()
