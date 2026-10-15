"""085: show every saved search result without changing research evidence."""

import ast
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.markers import MarkerStyle


IMPLEMENTATION = Path(__file__).resolve().parents[1]
SOURCE = IMPLEMENTATION / "reports/2026-10-14"
BUILDER = SOURCE / "build_preview.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evidence_hashes(root):
    """Hash source evidence and original plots, excluding executable code/cache."""
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".py" and "__pycache__" not in path.parts
    }


@pytest.fixture(scope="module")
def builder():
    specification = importlib.util.spec_from_file_location("search_visibility_preview", BUILDER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated(builder, tmp_path_factory):
    root = tmp_path_factory.mktemp("search-visibility") / "2026-10-15"
    before = evidence_hashes(SOURCE)
    receipt = builder.build_search_visibility(root)
    assert before == evidence_hashes(SOURCE), "Rendering must never overwrite source evidence."
    return root, receipt


@pytest.fixture
def isolated_source(builder, monkeypatch, tmp_path):
    root = tmp_path / "implementation/reports/2026-10-14"
    shutil.copytree(SOURCE, root, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(builder, "ROOT", root)
    return root


def test_report_leads_with_all_points_without_performance_or_run_id_tables(generated):
    output, receipt = generated
    text = (output / "README.md").read_text(encoding="utf-8")
    assert "search_capacity.png" in text and "search_capacity.svg" in text
    assert "search_results.csv" in text, "Keep exact values available as unchanged source data."
    assert not re.search(r"^\|", text, re.M), "The main report must be graph-first, not a run table."
    assert "All run IDs" not in text
    assert "Complete visible run tables" not in text
    assert "<details" not in text.lower()
    assert "validation" in text.lower() and "single" in text.lower()
    assert receipt["search_runs"] == 64
    assert receipt["architectures"] == 16
    assert receipt["runs_per_panel"] == 16
    assert receipt["plotted_experiments"] == list(range(1, 65))
    assert len(receipt["plot_keys"]) == len({tuple(key) for key in receipt["plot_keys"]}) == 64
    assert Counter(tuple(key[1:]) for key in receipt["plot_keys"]) == {
        (32, 8): 16, (32, 16): 16, (64, 8): 16, (64, 16): 16
    }


def test_annotations_select_real_best_worst_and_nearest_median_per_input_condition(generated):
    _, receipt = generated
    source = read_rows(SOURCE / "search_results.csv")
    annotations = receipt["annotations"]
    assert len(annotations) == 12
    for size, fps in ((32, 8), (32, 16), (64, 8), (64, 16)):
        group = [row for row in source
                 if (int(row["frame_size"]), int(row["fps"])) == (size, fps)]
        panel = [row for row in annotations if (row["frame_size"], row["fps"]) == (size, fps)]
        assert Counter(row["role"] for row in panel) == {"best": 1, "worst": 1, "median": 1}
        midpoint = median(float(row["accuracy"]) for row in group)
        expected = {
            "best": min(group, key=lambda row: (-float(row["accuracy"]), int(row["params"]),
                                                int(row["experiment"]))),
            "worst": min(group, key=lambda row: (float(row["accuracy"]), int(row["params"]),
                                                 int(row["experiment"]))),
            "median": min(group, key=lambda row: (round(abs(float(row["accuracy"]) - midpoint), 12),
                                                  int(row["params"]), int(row["experiment"]))),
        }
        for actual in panel:
            point = expected[actual["role"]]
            assert actual["experiment"] == int(point["experiment"])
            assert actual["architecture"] == point["architecture"]
            assert actual["accuracy"] == float(point["accuracy"])
    failed = [row for row in source if float(row["accuracy"]) == 0.125]
    assert len(failed) == 1
    assert int(failed[0]["experiment"]) in receipt["plotted_experiments"]
    assert any(row["role"] == "worst" and row["experiment"] == int(failed[0]["experiment"])
               for row in annotations)


def test_visibility_plot_has_four_panels_highlight_rings_and_extended_depth_legend(builder, monkeypatch):
    payload = json.loads((SOURCE / "preview_results.json").read_text(encoding="utf-8"))
    figures = []
    monkeypatch.setattr(builder, "save_plot", lambda figure, *args, **kwargs: figures.append(figure))
    receipt = builder.search_plot(payload["search"], payload["architecture_ranking"],
                                  annotate_examples=True)
    assert len(figures) == 1
    figure = figures[0]
    assert len(figure.axes) == 4, "No side run-ID tables or extra table axes."
    assert len(figure.legends) == 1, "The bottom legend shows layer counts only."
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "1 layer", "2 layers", "3 layers", "4 layers", "5 layers"
    ]
    assert not any("All run IDs" in text.get_text() for text in figure.findobj()
                   if hasattr(text, "get_text"))
    points = {}
    highlights = {}
    role_colors = {}
    depths = {row["experiment"]: row["layers"] for row in payload["search"]}
    depth_colors = {1: "#0072B2", 2: "#009E73", 3: "#D55E00", 4: "#CC79A7", 5: "#6A5ACD"}
    depth_markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    for axis in figure.axes:
        assert not axis.child_axes, "Compress low scores on the main axis, without zoom subplots."
        assert len(axis.collections) == 19, "Keep 16 observations and three hollow overlays per panel."
        arrows = [text for text in axis.texts if getattr(text, "arrow_patch", None) is not None]
        assert not arrows, "The per-panel key replaces arrows."
        panel_legend = axis.get_legend()
        assert panel_legend is not None
        labels = [text.get_text() for text in panel_legend.get_texts()]
        assert len(labels) == 3
        assert [label.split(":", 1)[0] for label in labels] == ["Best", "Middle", "Worst"]
        panel_points = set()
        for collection in axis.collections:
            identity = collection.get_gid()
            if identity.startswith("search-run-"):
                assert identity not in points
                points[identity] = collection.get_offsets()[0]
                experiment = int(identity.rsplit("-", 1)[1])
                panel_points.add(experiment)
                depth = depths[experiment]
                np.testing.assert_allclose(collection.get_facecolors()[0][:3], to_rgb(depth_colors[depth]))
                marker = MarkerStyle(depth_markers[depth])
                expected_path = marker.get_path().transformed(marker.get_transform())
                np.testing.assert_array_equal(collection.get_paths()[0].vertices, expected_path.vertices)
            else:
                assert re.fullmatch(r"example-(best|worst|median)-\d{2}", identity)
                assert identity not in highlights
                highlights[identity] = collection.get_offsets()[0]
                facecolors = collection.get_facecolors()
                assert not len(facecolors) or all(color[3] == 0 for color in facecolors)
                role = identity.split("-")[1]
                edgecolor = tuple(collection.get_edgecolors()[0])
                if role in role_colors:
                    assert role_colors[role] == edgecolor, "Role colors must agree across panels."
                role_colors[role] = edgecolor
        for row in receipt["annotations"]:
            if row["experiment"] in panel_points:
                role = "Middle" if row["role"] == "median" else row["role"].capitalize()
                assert f"{role}: {row['architecture']} ({100 * row['accuracy']:.2f}%)" in labels
        assert all("tied" not in label.lower() for label in labels)

        assert axis.get_yscale() == "function"
        scores = np.array([0, 12.5, 50, 89.9, 90, 92, 95, 100, 104])
        transform = axis.yaxis.get_transform()
        transformed = transform.transform(scores)
        expected = np.where(scores <= 90, scores * 0.05, 4.5 + scores - 90)
        np.testing.assert_allclose(transformed, expected)
        assert np.all(np.diff(transformed) > 0), "Scale must never invert or hide a score interval."
        np.testing.assert_allclose(transform.inverted().transform(transformed), scores)
        assert {0, 90, 100}.issubset(set(axis.get_yticks()))
        assert len([tick for tick in axis.get_yticks() if 90 <= tick <= 100]) >= 4
    assert set(points) == {f"search-run-{number:02d}" for number in range(1, 65)}
    for row in payload["search"]:
        x, y = points[f"search-run-{row['experiment']:02d}"]
        assert x == row["params"]
        assert y == pytest.approx(row["accuracy"] * 100)
    assert len(receipt["annotations"]) == 12
    assert len(highlights) == 12
    assert len(set(role_colors.values())) == 3
    for row in receipt["annotations"]:
        identity = f"example-{row['role']}-{row['experiment']:02d}"
        np.testing.assert_array_equal(highlights[identity], points[f"search-run-{row['experiment']:02d}"])
    assert receipt["y_axis"]["scale"] == "piecewise_linear"
    assert receipt["y_axis"]["threshold"] == 90
    assert receipt["y_axis"]["lower_compression"] == 0.05
    assert receipt["y_axis"]["note"], "Disclose the compressed lower interval."
    assert receipt["highlight_style"] == "ring_legend"
    assert receipt["zoom_panels"] == []


def test_plot_keeps_all_ids_and_full_data_bounds(generated):
    output, receipt = generated
    svg = ET.parse(output / "search_capacity.svg")
    ids = [element.attrib["id"] for element in svg.iter() if "id" in element.attrib]
    assert sorted(value for value in ids if value.startswith("search-run-")) == [
        f"search-run-{number:02d}" for number in range(1, 65)
    ]
    assert (output / "search_capacity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "search_capacity.png").stat().st_size > 10000
    assert len([value for value in ids if value.startswith("example-")]) == 12
    # A log-scale x axis cannot start at zero; every recorded point must remain inside.
    panels = receipt["axis_limits"]
    assert len(panels) == 4
    source = read_rows(SOURCE / "search_results.csv")
    for panel in panels:
        points = [row for row in source if (int(row["frame_size"]), int(row["fps"]))
                  == (panel["frame_size"], panel["fps"])]
        assert len(points) == 16
        for point in points:
            assert panel["x"][0] < int(point["params"]) < panel["x"][1]
            assert panel["y"][0] < float(point["accuracy"]) * 100 < panel["y"][1]


def test_all_original_image_links_are_visible_and_exist(generated):
    output, receipt = generated
    gallery = (output / "search_run_gallery.md").read_text(encoding="utf-8")
    assert "<details" not in gallery.lower()
    links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+\.png)\)", gallery)
    assert len(links) == len(set(links)) == 128
    assert receipt["original_images_linked"] == 128
    assert not (output / "search_images").exists(), "Link original evidence instead of copying it."
    resolved = {(output / link).resolve() for link in links}
    assert resolved == {path.resolve() for path in (SOURCE / "search_images").rglob("*.png")}
    assert all(path.is_file() for path in resolved)
    assert len(re.findall(r"^## Run \d{2}", gallery, re.M)) == 64


def test_receipt_reports_existing_hashes_without_claiming_new_run_audit(generated):
    output, returned = generated
    receipt = json.loads((output / "visibility_check.json").read_text(encoding="utf-8"))
    assert receipt == returned
    assert receipt["source_mode"] == "preserved_report"
    assert receipt["original_source_files_checked"] >= 0
    assert isinstance(receipt["original_source_files_unavailable"], list)
    original = json.loads((SOURCE / "search_evidence_check.json").read_text(encoding="utf-8"))
    assert receipt["original_source_files_checked"] + len(
        receipt["original_source_files_unavailable"]
    ) == len(original["source_sha256"])
    assert receipt["source_sha256"]
    for relative, expected in receipt["source_sha256"].items():
        assert digest((SOURCE / relative).resolve()) == expected
    assert digest(SOURCE / "colab_console.raw.txt") == (
        "c7609a5ec25cf93399de2f951ad93cb878212c99db88e3356da98d187e2cf98a"
    )


@pytest.mark.parametrize("change", ["score", "duplicate", "missing"])
def test_rejects_csv_numeric_changes_or_incomplete_coverage(builder, isolated_source, tmp_path, change):
    path = isolated_source / "search_results.csv"
    rows = read_rows(path)
    if change == "score":
        rows[0]["accuracy"] = "0.99"
    elif change == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows.pop()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "invalid-csv-output"
    with pytest.raises(ValueError):
        builder.build_search_visibility(output)
    assert not output.exists() or not any(output.iterdir())


def test_rejects_changed_portable_metrics(builder, isolated_source, tmp_path):
    path = isolated_source / "preview_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["search"][0]["artifact_validation_loss"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_search_visibility(tmp_path / "invalid-portable-output")


@pytest.mark.parametrize("change", ["changed", "missing"])
def test_rejects_corrupted_original_image(builder, isolated_source, tmp_path, change):
    image = isolated_source / "search_images/01/training_curves.png"
    if change == "changed":
        image.write_bytes(image.read_bytes() + b"changed")
    else:
        image.unlink()
    with pytest.raises(ValueError):
        builder.build_search_visibility(tmp_path / "invalid-image-output")


def test_never_overwrites_evidence_or_unrelated_output(builder, tmp_path):
    for unsafe in (SOURCE, SOURCE.parent, SOURCE / "new-report"):
        with pytest.raises(ValueError):
            builder.build_search_visibility(unsafe)
    output = tmp_path / "user-work"
    output.mkdir()
    note = output / "notes.txt"
    note.write_text("Keep this user work.", encoding="utf-8")
    with pytest.raises(ValueError):
        builder.build_search_visibility(output)
    assert note.read_text(encoding="utf-8") == "Keep this user work."
    assert list(output.iterdir()) == [note]


def test_unchanged_report_can_be_rebuilt_without_numeric_or_export_drift(builder, generated):
    output, _ = generated
    before = evidence_hashes(output)
    receipt = builder.build_search_visibility(output)
    assert evidence_hashes(output) == before
    assert receipt["plotted_experiments"] == list(range(1, 65))


def test_edited_generated_report_is_preserved(builder, generated, tmp_path):
    output = tmp_path / "edited-report"
    shutil.copytree(generated[0], output)
    readme = output / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nUser review notes.\n", encoding="utf-8")
    before = evidence_hashes(output)
    with pytest.raises(ValueError, match="edited"):
        builder.build_search_visibility(output)
    assert evidence_hashes(output) == before
    assert readme.read_text(encoding="utf-8").endswith("User review notes.\n")


def test_file_cannot_be_used_as_output_directory(builder, tmp_path):
    output = tmp_path / "user-notes.md"
    output.write_text("Preserve this file.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        builder.build_search_visibility(output)
    assert output.read_text(encoding="utf-8") == "Preserve this file.\n"


def test_legacy_full_review_plot_entry_still_renders_both_original_figures(builder, monkeypatch):
    payload = json.loads((SOURCE / "preview_results.json").read_text(encoding="utf-8"))
    before = evidence_hashes(SOURCE)
    figures = {}

    def capture(fig, stem, output_dir=None):
        figures[stem] = fig

    monkeypatch.setattr(builder, "save_plot", capture)
    builder.plots(payload["search"], payload["architecture_ranking"], payload["comparison"])
    assert list(figures) == ["search_capacity", "comparison_validation"]
    assert len(figures["search_capacity"].axes) == 4
    assert len(figures["comparison_validation"].axes) == 2
    points = {}
    for axis in figures["search_capacity"].axes:
        assert axis.get_xlim() == (35000, 4500000)
        assert axis.get_ylim() == (0, 104)
        assert len(axis.collections) == 16
        for collection in axis.collections:
            assert collection.get_gid() not in points
            points[collection.get_gid()] = collection.get_offsets()[0]
    assert set(points) == {f"search-run-{number:02d}" for number in range(1, 65)}
    for row in payload["search"]:
        x, y = points[f"search-run-{row['experiment']:02d}"]
        assert x == row["params"]
        assert y == pytest.approx(row["accuracy"] * 100)
    assert evidence_hashes(SOURCE) == before


def test_builder_does_not_import_training_or_video_modules():
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.split(".")[0] in {"src", "torch", "torchvision", "subprocess"}
                   for name in imports)
