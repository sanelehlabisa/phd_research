Already up to date.
Code checkout: /content/phd_research @ bc3d8fb
Notebook Python: /usr/bin/python3
{'python': '3.13.15', 'platform': 'Linux-6.6.122+-x86_64-with-glibc2.39', 'pytorch': '2.9.1+cu128', 'torchvision': '0.24.1+cu128', 'cuda_available': True, 'cuda_version': '12.8', 'cudnn_version': 91002, 'cuda_device': 'NVIDIA A100-SXM4-80GB'}
1/2 Focused search, confirmation and ablations (validation only)
$ /usr/bin/python3 -m src.experiments --config /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/configs/experiments/aad_wide_depth_search_colab.json --list-plan
Ticket 075: 32 wide 1-5-layer architectures; 64 base custom runs; test locked
Evidence-guided, not exhaustive or unbiased; user-reviewed filename independence, not proven subject independence.
calibration: up to 12 jobs before exact-config reuse
flat: up to 40 jobs before exact-config reuse
references: up to 4 jobs before exact-config reuse
refinement: up to 24 jobs before exact-config reuse
confirmation: up to 24 jobs before exact-config reuse
weight_decay: up to 12 jobs before exact-config reuse
temporal: up to 15 jobs before exact-config reuse
Upper budget: 131 search jobs + up to 3 recipe checks + 8 final trainings = 142; 256 search epochs cap, minimum 64, patience 24.
Automatic notebook: then eight comparison trainings, 512 cap/minimum 128/patience 32; frozen-test guard remains.
custom: custom_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_128 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_96_96 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_128_128 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32_32_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_96_96_96 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_128_128_128 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32_32_32_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_96_96_96_96 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_128_128_128_128 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_8_8_8 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_16_16_16 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_48_48_48 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_80_80_80 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_128_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_128_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_128 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_32_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_96_64_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_96_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_64_96 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32_64 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_64_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_16_32_32 | 32x32 | 8 frames/16 FPS | seed=42
custom: custom_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_128 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_32_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_96_96 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_128_128 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_32_32_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_96_96_96 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_128_128_128 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_32_32_32_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_96_96_96_96 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_128_128_128_128 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_8_8_8 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_16_16_16 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_48_48_48 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_80_80_80 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_64_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_128_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_128_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_128 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_32_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_32_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_96_64_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_96_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_64_96 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_32_64 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_64_32 | 64x64 | 8 frames/16 FPS | seed=42
custom: custom_16_32_32 | 64x64 | 8 frames/16 FPS | seed=42
$ /usr/bin/python3 -m src.experiments --config /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/notebook_studies/capacity_20261010_144521_885468/resolved_capacity_search.json
Downloading AAD from Kaggle: sanelehlabisa/abnormal-activities-dataset
Downloading to /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/2.archive...
100%|██████████| 59.7M/59.7M [00:01<00:00, 38.4MB/s]
Extracting files...
AAD dataset ready: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Experiment 1/142 campaign maximum | calibration: custom_64_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_144553_812238_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Caching 908 train/validation clips (test excluded)...
  Cached 50/908
  Cached 100/908
  Cached 150/908
  Cached 200/908
  Cached 250/908
  Cached 300/908
  Cached 350/908
  Cached 400/908
  Cached 450/908
  Cached 500/908
  Cached 550/908
  Cached 600/908
  Cached 650/908
  Cached 700/908
  Cached 750/908
  Cached 800/908
  Cached 850/908
  Cached 900/908
  Cached 908/908
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 1/142 maximum | calibration

Experiment 1/142 maximum | calibration: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_144553_812238_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_144553_812238_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_144553_812238_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3310 selected_epoch=249 time=806s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_144553_812238_abnormal-activities-dataset_capacity/summary.json
Experiment 2/142 campaign maximum | calibration: custom_64_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150008_190307_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 2/142 maximum | calibration

Experiment 2/142 maximum | calibration: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150008_190307_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150008_190307_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150008_190307_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9438 val_loss=0.4347 selected_epoch=128 time=476s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.8, 'recall': 0.9230769230769231, 'f1': 0.8571428571428571, 'support': 13, 'tp': 12, 'fp': 3, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9230769230769231, 'recall': 0.8, 'f1': 0.8571428571428571, 'support': 15, 'tp': 12, 'fp': 1, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150008_190307_abnormal-activities-dataset_capacity/summary.json
Experiment 3/142 campaign maximum | calibration: custom_64_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150807_770978_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 3/142 maximum | calibration

Experiment 3/142 maximum | calibration: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150807_770978_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150807_770978_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150807_770978_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.4312 selected_epoch=172 time=613s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_150807_770978_abnormal-activities-dataset_capacity/summary.json
Experiment 4/142 campaign maximum | calibration: custom_128_128_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_151825_073241_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 4/142 maximum | calibration

Experiment 4/142 maximum | calibration: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_151825_073241_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_151825_073241_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_151825_073241_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.3066 selected_epoch=179 time=1574s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_151825_073241_abnormal-activities-dataset_capacity/summary.json
Experiment 5/142 campaign maximum | calibration: custom_128_128_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_154445_758554_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 5/142 maximum | calibration

Experiment 5/142 maximum | calibration: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_154445_758554_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_154445_758554_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_154445_758554_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.2586 selected_epoch=223 time=1914s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8571428571428571, 'recall': 0.9230769230769231, 'f1': 0.8888888888888888, 'support': 13, 'tp': 12, 'fp': 2, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_154445_758554_abnormal-activities-dataset_capacity/summary.json
Experiment 6/142 campaign maximum | calibration: custom_128_128_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_161645_833250_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 6/142 maximum | calibration

Experiment 6/142 maximum | calibration: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_161645_833250_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_161645_833250_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_161645_833250_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.5481 selected_epoch=190 time=1649s
{'class_name': 'Begging', 'precision': 0.9444444444444444, 'recall': 0.85, 'f1': 0.8947368421052632, 'support': 20, 'tp': 17, 'fp': 1, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0.9166666666666666, 'recall': 0.8461538461538461, 'f1': 0.88, 'support': 13, 'tp': 11, 'fp': 1, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.875, 'recall': 0.9333333333333333, 'f1': 0.9032258064516129, 'support': 15, 'tp': 14, 'fp': 2, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_161645_833250_abnormal-activities-dataset_capacity/summary.json
Experiment 7/142 campaign maximum | calibration: r3d_18 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164420_957320_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 7/142 maximum | calibration

Experiment 7/142 maximum | calibration: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164420_957320_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164420_957320_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164420_957320_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9312 val_loss=0.2763 selected_epoch=31 time=80s
{'class_name': 'Begging', 'precision': 0.9444444444444444, 'recall': 0.85, 'f1': 0.8947368421052632, 'support': 20, 'tp': 17, 'fp': 1, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0.8571428571428571, 'recall': 0.9230769230769231, 'f1': 0.8888888888888888, 'support': 13, 'tp': 12, 'fp': 2, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.8666666666666667, 'recall': 0.9285714285714286, 'f1': 0.896551724137931, 'support': 14, 'tp': 13, 'fp': 2, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164420_957320_abnormal-activities-dataset_capacity/summary.json
Experiment 8/142 campaign maximum | calibration: r3d_18 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164549_881921_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 8/142 maximum | calibration

Experiment 8/142 maximum | calibration: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164549_881921_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164549_881921_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164549_881921_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9500 val_loss=0.2591 selected_epoch=61 time=105s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.8571428571428571, 'recall': 0.8571428571428571, 'f1': 0.8571428571428571, 'support': 14, 'tp': 12, 'fp': 2, 'fn': 2}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 1.0, 'recall': 0.9333333333333333, 'f1': 0.9655172413793104, 'support': 15, 'tp': 14, 'fp': 0, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.8095238095238095, 'recall': 1.0, 'f1': 0.8947368421052632, 'support': 17, 'tp': 17, 'fp': 4, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164549_881921_abnormal-activities-dataset_capacity/summary.json
Experiment 9/142 campaign maximum | calibration: r3d_18 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164743_787501_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 9/142 maximum | calibration

Experiment 9/142 maximum | calibration: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164743_787501_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164743_787501_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164743_787501_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9000 val_loss=0.3316 selected_epoch=35 time=79s
{'class_name': 'Begging', 'precision': 0.9375, 'recall': 0.75, 'f1': 0.8333333333333334, 'support': 20, 'tp': 15, 'fp': 1, 'fn': 5}
{'class_name': 'Drunkenness', 'precision': 0.8666666666666667, 'recall': 1.0, 'f1': 0.9285714285714286, 'support': 13, 'tp': 13, 'fp': 2, 'fn': 0}
{'class_name': 'Fight', 'precision': 0.75, 'recall': 0.8571428571428571, 'f1': 0.8, 'support': 14, 'tp': 12, 'fp': 4, 'fn': 2}
{'class_name': 'Harassment', 'precision': 0.9166666666666666, 'recall': 0.7857142857142857, 'f1': 0.8461538461538461, 'support': 14, 'tp': 11, 'fp': 1, 'fn': 3}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.8095238095238095, 'recall': 1.0, 'f1': 0.8947368421052632, 'support': 17, 'tp': 17, 'fp': 4, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8235294117647058, 'f1': 0.9032258064516129, 'support': 17, 'tp': 14, 'fp': 0, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164743_787501_abnormal-activities-dataset_capacity/summary.json
Experiment 10/142 campaign maximum | calibration: swin3d_t 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164911_804295_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 10/142 maximum | calibration

Experiment 10/142 maximum | calibration: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164911_804295_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164911_804295_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164911_804295_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3843 selected_epoch=102 time=302s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_164911_804295_abnormal-activities-dataset_capacity/summary.json
Experiment 11/142 campaign maximum | calibration: swin3d_t 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165423_733595_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 11/142 maximum | calibration

Experiment 11/142 maximum | calibration: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165423_733595_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165423_733595_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165423_733595_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=60 time=207s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165423_733595_abnormal-activities-dataset_capacity/summary.json
Experiment 12/142 campaign maximum | calibration: swin3d_t 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165800_413051_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 12/142 maximum | calibration

Experiment 12/142 maximum | calibration: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165800_413051_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165800_413051_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165800_413051_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=60 time=210s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_165800_413051_abnormal-activities-dataset_capacity/summary.json

Full results: 12 completed; 0 failed/pending
  1 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
  2 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
  3 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
  4 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
  5 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
  6 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
  7 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
  8 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
  9 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 10 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 11 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
 12 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): flat, references, refinement, confirmation, weight_decay, temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (1,819,212,804 bytes; 307 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 13/142 campaign maximum | flat: custom_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170340_222246_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Caching 908 train/validation clips (test excluded)...
  Cached 50/908
  Cached 100/908
  Cached 150/908
  Cached 200/908
  Cached 250/908
  Cached 300/908
  Cached 350/908
  Cached 400/908
  Cached 450/908
  Cached 500/908
  Cached 550/908
  Cached 600/908
  Cached 650/908
  Cached 700/908
  Cached 750/908
  Cached 800/908
  Cached 850/908
  Cached 900/908
  Cached 908/908
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 13/142 maximum | flat

Experiment 13/142 maximum | flat: custom_32 | params=40,811
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170340_222246_abnormal-activities-dataset_capacity/models/custom_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170340_222246_abnormal-activities-dataset_capacity/models/custom_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170340_222246_abnormal-activities-dataset_capacity/models/custom_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8500 val_loss=0.7458 selected_epoch=254 time=143s
{'class_name': 'Begging', 'precision': 0.8571428571428571, 'recall': 0.9, 'f1': 0.8780487804878049, 'support': 20, 'tp': 18, 'fp': 3, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.9166666666666666, 'recall': 0.8461538461538461, 'f1': 0.88, 'support': 13, 'tp': 11, 'fp': 1, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.8125, 'recall': 0.9285714285714286, 'f1': 0.8666666666666667, 'support': 14, 'tp': 13, 'fp': 3, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.7857142857142857, 'recall': 0.7857142857142857, 'f1': 0.7857142857142857, 'support': 14, 'tp': 11, 'fp': 3, 'fn': 3}
{'class_name': 'Hijack', 'precision': 0.7272727272727273, 'recall': 0.6666666666666666, 'f1': 0.6956521739130435, 'support': 12, 'tp': 8, 'fp': 3, 'fn': 4}
{'class_name': 'Knife Hazard', 'precision': 0.8333333333333334, 'recall': 0.6666666666666666, 'f1': 0.7407407407407407, 'support': 15, 'tp': 10, 'fp': 2, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.8095238095238095, 'recall': 1.0, 'f1': 0.8947368421052632, 'support': 17, 'tp': 17, 'fp': 4, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9090909090909091, 'recall': 0.7692307692307693, 'f1': 0.8333333333333334, 'support': 13, 'tp': 10, 'fp': 1, 'fn': 3}
{'class_name': 'Terrorism', 'precision': 0.6923076923076923, 'recall': 0.8181818181818182, 'f1': 0.75, 'support': 11, 'tp': 9, 'fp': 4, 'fn': 2}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170340_222246_abnormal-activities-dataset_capacity/summary.json
Experiment 14/142 campaign maximum | flat: custom_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170616_911100_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 14/142 maximum | flat

Experiment 14/142 maximum | flat: custom_64 | params=155,339
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170616_911100_abnormal-activities-dataset_capacity/models/custom_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170616_911100_abnormal-activities-dataset_capacity/models/custom_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170616_911100_abnormal-activities-dataset_capacity/models/custom_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.3708 selected_epoch=251 time=152s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170616_911100_abnormal-activities-dataset_capacity/summary.json
Experiment 15/142 campaign maximum | flat: custom_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170851_628347_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 15/142 maximum | flat

Experiment 15/142 maximum | flat: custom_128 | params=605,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170851_628347_abnormal-activities-dataset_capacity/models/custom_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170851_628347_abnormal-activities-dataset_capacity/models/custom_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170851_628347_abnormal-activities-dataset_capacity/models/custom_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3479 selected_epoch=216 time=222s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_170851_628347_abnormal-activities-dataset_capacity/summary.json
Experiment 16/142 campaign maximum | flat: custom_32_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171236_817115_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 16/142 maximum | flat

Experiment 16/142 maximum | flat: custom_32_32 | params=114,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171236_817115_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171236_817115_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171236_817115_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8875 val_loss=0.5074 selected_epoch=255 time=187s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6666666666666666, 'recall': 0.9230769230769231, 'f1': 0.7741935483870968, 'support': 13, 'tp': 12, 'fp': 6, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.7857142857142857, 'recall': 0.7857142857142857, 'f1': 0.7857142857142857, 'support': 14, 'tp': 11, 'fp': 3, 'fn': 3}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9090909090909091, 'recall': 0.6666666666666666, 'f1': 0.7692307692307693, 'support': 15, 'tp': 10, 'fp': 1, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.8823529411764706, 'recall': 0.8823529411764706, 'f1': 0.8823529411764706, 'support': 17, 'tp': 15, 'fp': 2, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.7894736842105263, 'recall': 0.8823529411764706, 'f1': 0.8333333333333334, 'support': 17, 'tp': 15, 'fp': 4, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9090909090909091, 'recall': 0.7692307692307693, 'f1': 0.8333333333333334, 'support': 13, 'tp': 10, 'fp': 1, 'fn': 3}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171236_817115_abnormal-activities-dataset_capacity/summary.json
Experiment 17/142 campaign maximum | flat: custom_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171546_575056_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 17/142 maximum | flat

Experiment 17/142 maximum | flat: custom_64_64 | params=450,507
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171546_575056_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171546_575056_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171546_575056_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2161 selected_epoch=193 time=186s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8461538461538461, 'recall': 0.9166666666666666, 'f1': 0.88, 'support': 12, 'tp': 11, 'fp': 2, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 11, 'tp': 10, 'fp': 1, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171546_575056_abnormal-activities-dataset_capacity/summary.json
Experiment 18/142 campaign maximum | flat: custom_96_96 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171855_831360_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 18/142 maximum | flat

Experiment 18/142 maximum | flat: custom_96_96 | params=1,007,531
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171855_831360_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171855_831360_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171855_831360_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2729 selected_epoch=149 time=208s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_171855_831360_abnormal-activities-dataset_capacity/summary.json
Experiment 19/142 campaign maximum | flat: custom_128_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172227_567886_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 19/142 maximum | flat

Experiment 19/142 maximum | flat: custom_128_128 | params=1,785,739
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172227_567886_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172227_567886_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172227_567886_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1666 selected_epoch=149 time=276s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172227_567886_abnormal-activities-dataset_capacity/summary.json
Experiment 20/142 campaign maximum | flat: custom_32_32_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172707_032353_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 20/142 maximum | flat

Experiment 20/142 maximum | flat: custom_32_32_32 | params=188,523
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172707_032353_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172707_032353_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172707_032353_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.2566 selected_epoch=250 time=224s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.7647058823529411, 'recall': 1.0, 'f1': 0.8666666666666667, 'support': 13, 'tp': 13, 'fp': 4, 'fn': 0}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 1.0, 'recall': 0.7333333333333333, 'f1': 0.8461538461538461, 'support': 15, 'tp': 11, 'fp': 0, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_172707_032353_abnormal-activities-dataset_capacity/summary.json
Experiment 21/142 campaign maximum | flat: custom_64_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173053_781632_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 21/142 maximum | flat

Experiment 21/142 maximum | flat: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173053_781632_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173053_781632_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173053_781632_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2176 selected_epoch=182 time=235s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173053_781632_abnormal-activities-dataset_capacity/summary.json
Experiment 22/142 campaign maximum | flat: custom_96_96_96 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173451_612878_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 22/142 maximum | flat

Experiment 22/142 maximum | flat: custom_96_96_96 | params=1,671,467
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173451_612878_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173451_612878_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173451_612878_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1761 selected_epoch=123 time=253s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173451_612878_abnormal-activities-dataset_capacity/summary.json
Experiment 23/142 campaign maximum | flat: custom_128_128_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173908_251646_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 23/142 maximum | flat

Experiment 23/142 maximum | flat: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173908_251646_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173908_251646_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173908_251646_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.1994 selected_epoch=162 time=428s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_173908_251646_abnormal-activities-dataset_capacity/summary.json
Experiment 24/142 campaign maximum | flat: custom_32_32_32_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_174620_666692_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 24/142 maximum | flat

Experiment 24/142 maximum | flat: custom_32_32_32_32 | params=262,379
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_174620_666692_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_174620_666692_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_174620_666692_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2135 selected_epoch=221 time=261s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_174620_666692_abnormal-activities-dataset_capacity/summary.json
Experiment 25/142 campaign maximum | flat: custom_64_64_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175045_081749_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 25/142 maximum | flat

Experiment 25/142 maximum | flat: custom_64_64_64_64 | params=1,040,843
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175045_081749_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175045_081749_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175045_081749_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1458 selected_epoch=173 time=278s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175045_081749_abnormal-activities-dataset_capacity/summary.json
Experiment 26/142 campaign maximum | flat: custom_96_96_96_96 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175526_771936_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 26/142 maximum | flat

Experiment 26/142 maximum | flat: custom_96_96_96_96 | params=2,335,403
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175526_771936_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175526_771936_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175526_771936_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.2095 selected_epoch=194 time=481s
{'class_name': 'Begging', 'precision': 0.9473684210526315, 'recall': 0.9, 'f1': 0.9230769230769231, 'support': 20, 'tp': 18, 'fp': 1, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.8823529411764706, 'recall': 1.0, 'f1': 0.9375, 'support': 15, 'tp': 15, 'fp': 2, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_175526_771936_abnormal-activities-dataset_capacity/summary.json
Experiment 27/142 campaign maximum | flat: custom_128_128_128_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_180331_973324_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 27/142 maximum | flat

Experiment 27/142 maximum | flat: custom_128_128_128_128 | params=4,146,059
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_180331_973324_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_180331_973324_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_180331_973324_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2164 selected_epoch=175 time=594s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_180331_973324_abnormal-activities-dataset_capacity/summary.json
Experiment 28/142 campaign maximum | flat: custom_8_8_8 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181330_838764_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 28/142 maximum | flat

Experiment 28/142 maximum | flat: custom_8_8_8 | params=12,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181330_838764_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181330_838764_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181330_838764_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.5375 val_loss=1.1988 selected_epoch=254 time=221s
{'class_name': 'Begging', 'precision': 0.5483870967741935, 'recall': 0.85, 'f1': 0.6666666666666666, 'support': 20, 'tp': 17, 'fp': 14, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0.625, 'recall': 0.4166666666666667, 'f1': 0.5, 'support': 12, 'tp': 5, 'fp': 3, 'fn': 7}
{'class_name': 'Knife Hazard', 'precision': 0.36666666666666664, 'recall': 0.7333333333333333, 'f1': 0.4888888888888889, 'support': 15, 'tp': 11, 'fp': 19, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.3684210526315789, 'recall': 0.8235294117647058, 'f1': 0.509090909090909, 'support': 17, 'tp': 14, 'fp': 24, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.3333333333333333, 'recall': 0.15384615384615385, 'f1': 0.21052631578947367, 'support': 13, 'tp': 2, 'fp': 4, 'fn': 11}
{'class_name': 'Terrorism', 'precision': 0.5333333333333333, 'recall': 0.7272727272727273, 'f1': 0.6153846153846154, 'support': 11, 'tp': 8, 'fp': 7, 'fn': 3}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181330_838764_abnormal-activities-dataset_capacity/summary.json
Experiment 29/142 campaign maximum | flat: custom_16_16_16 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181714_773457_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 29/142 maximum | flat

Experiment 29/142 maximum | flat: custom_16_16_16 | params=48,187
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181714_773457_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181714_773457_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181714_773457_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8250 val_loss=0.6872 selected_epoch=255 time=228s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6, 'recall': 0.9230769230769231, 'f1': 0.7272727272727273, 'support': 13, 'tp': 12, 'fp': 8, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.6666666666666666, 'recall': 0.42857142857142855, 'f1': 0.5217391304347826, 'support': 14, 'tp': 6, 'fp': 3, 'fn': 8}
{'class_name': 'Harassment', 'precision': 0.8888888888888888, 'recall': 0.5714285714285714, 'f1': 0.6956521739130435, 'support': 14, 'tp': 8, 'fp': 1, 'fn': 6}
{'class_name': 'Hijack', 'precision': 0.8888888888888888, 'recall': 0.6666666666666666, 'f1': 0.7619047619047619, 'support': 12, 'tp': 8, 'fp': 1, 'fn': 4}
{'class_name': 'Knife Hazard', 'precision': 0.8461538461538461, 'recall': 0.7333333333333333, 'f1': 0.7857142857142857, 'support': 15, 'tp': 11, 'fp': 2, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.7142857142857143, 'recall': 0.8823529411764706, 'f1': 0.7894736842105263, 'support': 17, 'tp': 15, 'fp': 6, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.7333333333333333, 'recall': 1.0, 'f1': 0.8461538461538461, 'support': 11, 'tp': 11, 'fp': 4, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_181714_773457_abnormal-activities-dataset_capacity/summary.json
Experiment 30/142 campaign maximum | flat: custom_48_48_48 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182106_040470_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 30/142 maximum | flat

Experiment 30/142 maximum | flat: custom_48_48_48 | params=421,019
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182106_040470_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182106_040470_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182106_040470_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1894 selected_epoch=171 time=196s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182106_040470_abnormal-activities-dataset_capacity/summary.json
Experiment 31/142 campaign maximum | flat: custom_80_80_80 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182424_918498_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 31/142 maximum | flat

Experiment 31/142 maximum | flat: custom_80_80_80 | params=1,162,491
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182424_918498_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182424_918498_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182424_918498_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2379 selected_epoch=140 time=238s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182424_918498_abnormal-activities-dataset_capacity/summary.json
Experiment 32/142 campaign maximum | flat: custom_64_64_64_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182826_206520_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 32/142 maximum | flat

Experiment 32/142 maximum | flat: custom_64_64_64_64_64 | params=1,336,011
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182826_206520_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182826_206520_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182826_206520_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1678 selected_epoch=195 time=370s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_182826_206520_abnormal-activities-dataset_capacity/summary.json
Experiment 33/142 campaign maximum | flat: custom_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183439_627905_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 33/142 maximum | flat

Experiment 33/142 maximum | flat: custom_32 | params=40,811
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183439_627905_abnormal-activities-dataset_capacity/models/custom_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183439_627905_abnormal-activities-dataset_capacity/models/custom_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183439_627905_abnormal-activities-dataset_capacity/models/custom_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.5688 val_loss=1.1316 selected_epoch=245 time=194s
{'class_name': 'Begging', 'precision': 0.42857142857142855, 'recall': 0.75, 'f1': 0.5454545454545454, 'support': 20, 'tp': 15, 'fp': 20, 'fn': 5}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 1, 'fn': 14}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.14285714285714285, 'f1': 0.25, 'support': 14, 'tp': 2, 'fp': 0, 'fn': 12}
{'class_name': 'Hijack', 'precision': 0.5, 'recall': 0.4166666666666667, 'f1': 0.45454545454545453, 'support': 12, 'tp': 5, 'fp': 5, 'fn': 7}
{'class_name': 'Knife Hazard', 'precision': 0.4, 'recall': 0.6666666666666666, 'f1': 0.5, 'support': 15, 'tp': 10, 'fp': 15, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.7142857142857143, 'recall': 0.8823529411764706, 'f1': 0.7894736842105263, 'support': 17, 'tp': 15, 'fp': 6, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.5454545454545454, 'recall': 0.7058823529411765, 'f1': 0.6153846153846154, 'support': 17, 'tp': 12, 'fp': 10, 'fn': 5}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.7058823529411765, 'recall': 0.9230769230769231, 'f1': 0.8, 'support': 13, 'tp': 12, 'fp': 5, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.46153846153846156, 'recall': 0.5454545454545454, 'f1': 0.5, 'support': 11, 'tp': 6, 'fp': 7, 'fn': 5}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183439_627905_abnormal-activities-dataset_capacity/summary.json
Experiment 34/142 campaign maximum | flat: custom_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183758_516242_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 34/142 maximum | flat

Experiment 34/142 maximum | flat: custom_64 | params=155,339
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183758_516242_abnormal-activities-dataset_capacity/models/custom_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183758_516242_abnormal-activities-dataset_capacity/models/custom_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183758_516242_abnormal-activities-dataset_capacity/models/custom_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8188 val_loss=0.7404 selected_epoch=256 time=290s
{'class_name': 'Begging', 'precision': 0.7916666666666666, 'recall': 0.95, 'f1': 0.8636363636363636, 'support': 20, 'tp': 19, 'fp': 5, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6666666666666666, 'recall': 0.6153846153846154, 'f1': 0.64, 'support': 13, 'tp': 8, 'fp': 4, 'fn': 5}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.6428571428571429, 'f1': 0.782608695652174, 'support': 14, 'tp': 9, 'fp': 0, 'fn': 5}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.5714285714285714, 'f1': 0.7272727272727273, 'support': 14, 'tp': 8, 'fp': 0, 'fn': 6}
{'class_name': 'Hijack', 'precision': 0.9, 'recall': 0.75, 'f1': 0.8181818181818182, 'support': 12, 'tp': 9, 'fp': 1, 'fn': 3}
{'class_name': 'Knife Hazard', 'precision': 0.6875, 'recall': 0.7333333333333333, 'f1': 0.7096774193548387, 'support': 15, 'tp': 11, 'fp': 5, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.7368421052631579, 'recall': 0.8235294117647058, 'f1': 0.7777777777777778, 'support': 17, 'tp': 14, 'fp': 5, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.8125, 'recall': 1.0, 'f1': 0.896551724137931, 'support': 13, 'tp': 13, 'fp': 3, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.7857142857142857, 'recall': 1.0, 'f1': 0.88, 'support': 11, 'tp': 11, 'fp': 3, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_183758_516242_abnormal-activities-dataset_capacity/summary.json
Experiment 35/142 campaign maximum | flat: custom_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_184252_102262_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 35/142 maximum | flat

Experiment 35/142 maximum | flat: custom_128 | params=605,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_184252_102262_abnormal-activities-dataset_capacity/models/custom_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_184252_102262_abnormal-activities-dataset_capacity/models/custom_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_184252_102262_abnormal-activities-dataset_capacity/models/custom_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.3872 selected_epoch=249 time=647s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_184252_102262_abnormal-activities-dataset_capacity/summary.json
Experiment 36/142 campaign maximum | flat: custom_32_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185342_875760_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 36/142 maximum | flat

Experiment 36/142 maximum | flat: custom_32_32 | params=114,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185342_875760_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185342_875760_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185342_875760_abnormal-activities-dataset_capacity/models/custom_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8062 val_loss=0.8101 selected_epoch=256 time=312s
{'class_name': 'Begging', 'precision': 0.7037037037037037, 'recall': 0.95, 'f1': 0.8085106382978723, 'support': 20, 'tp': 19, 'fp': 8, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.6153846153846154, 'f1': 0.7619047619047619, 'support': 13, 'tp': 8, 'fp': 0, 'fn': 5}
{'class_name': 'Fight', 'precision': 0.9166666666666666, 'recall': 0.7857142857142857, 'f1': 0.8461538461538461, 'support': 14, 'tp': 11, 'fp': 1, 'fn': 3}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.5714285714285714, 'f1': 0.7272727272727273, 'support': 14, 'tp': 8, 'fp': 0, 'fn': 6}
{'class_name': 'Hijack', 'precision': 0.7, 'recall': 0.5833333333333334, 'f1': 0.6363636363636364, 'support': 12, 'tp': 7, 'fp': 3, 'fn': 5}
{'class_name': 'Knife Hazard', 'precision': 0.7142857142857143, 'recall': 0.6666666666666666, 'f1': 0.6896551724137931, 'support': 15, 'tp': 10, 'fp': 4, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.8823529411764706, 'recall': 0.8823529411764706, 'f1': 0.8823529411764706, 'support': 17, 'tp': 15, 'fp': 2, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.7222222222222222, 'recall': 1.0, 'f1': 0.8387096774193549, 'support': 13, 'tp': 13, 'fp': 5, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.6428571428571429, 'recall': 0.8181818181818182, 'f1': 0.72, 'support': 11, 'tp': 9, 'fp': 5, 'fn': 2}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185342_875760_abnormal-activities-dataset_capacity/summary.json
Experiment 37/142 campaign maximum | flat: custom_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185858_290222_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 37/142 maximum | flat

Experiment 37/142 maximum | flat: custom_64_64 | params=450,507
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185858_290222_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185858_290222_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185858_290222_abnormal-activities-dataset_capacity/models/custom_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.3930 selected_epoch=254 time=549s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_185858_290222_abnormal-activities-dataset_capacity/summary.json
Experiment 38/142 campaign maximum | flat: custom_96_96 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_190811_594591_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 38/142 maximum | flat

Experiment 38/142 maximum | flat: custom_96_96 | params=1,007,531
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_190811_594591_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_190811_594591_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_190811_594591_abnormal-activities-dataset_capacity/models/custom_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3620 selected_epoch=222 time=938s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_190811_594591_abnormal-activities-dataset_capacity/summary.json
Experiment 39/142 campaign maximum | flat: custom_128_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_192353_830757_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 39/142 maximum | flat

Experiment 39/142 maximum | flat: custom_128_128 | params=1,785,739
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_192353_830757_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_192353_830757_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_192353_830757_abnormal-activities-dataset_capacity/models/custom_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.3569 selected_epoch=250 time=1320s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_192353_830757_abnormal-activities-dataset_capacity/summary.json
Experiment 40/142 campaign maximum | flat: custom_32_32_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_194559_719470_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 40/142 maximum | flat

Experiment 40/142 maximum | flat: custom_32_32_32 | params=188,523
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_194559_719470_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_194559_719470_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_194559_719470_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8438 val_loss=0.6195 selected_epoch=255 time=431s
{'class_name': 'Begging', 'precision': 0.8636363636363636, 'recall': 0.95, 'f1': 0.9047619047619048, 'support': 20, 'tp': 19, 'fp': 3, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.75, 'recall': 0.9230769230769231, 'f1': 0.8275862068965517, 'support': 13, 'tp': 12, 'fp': 4, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.5714285714285714, 'recall': 0.5714285714285714, 'f1': 0.5714285714285714, 'support': 14, 'tp': 8, 'fp': 6, 'fn': 6}
{'class_name': 'Harassment', 'precision': 0.8461538461538461, 'recall': 0.7857142857142857, 'f1': 0.8148148148148148, 'support': 14, 'tp': 11, 'fp': 2, 'fn': 3}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.8888888888888888, 'recall': 0.5333333333333333, 'f1': 0.6666666666666666, 'support': 15, 'tp': 8, 'fp': 1, 'fn': 7}
{'class_name': 'Normal Videos', 'precision': 0.8823529411764706, 'recall': 0.8823529411764706, 'f1': 0.8823529411764706, 'support': 17, 'tp': 15, 'fp': 2, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.75, 'recall': 0.8823529411764706, 'f1': 0.8108108108108109, 'support': 17, 'tp': 15, 'fp': 5, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_194559_719470_abnormal-activities-dataset_capacity/summary.json
Experiment 1/142 maximum | flat custom_64_64_64 (verified reuse)
Experiment 41/142 campaign maximum | flat: custom_96_96_96 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_195314_950124_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 41/142 maximum | flat

Experiment 41/142 maximum | flat: custom_96_96_96 | params=1,671,467
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_195314_950124_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_195314_950124_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_195314_950124_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9563 val_loss=0.4163 selected_epoch=219 time=1427s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8333333333333334, 'recall': 0.7692307692307693, 'f1': 0.8, 'support': 13, 'tp': 10, 'fp': 2, 'fn': 3}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.8125, 'recall': 0.8666666666666667, 'f1': 0.8387096774193549, 'support': 15, 'tp': 13, 'fp': 3, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_195314_950124_abnormal-activities-dataset_capacity/summary.json
Experiment 4/142 maximum | flat custom_128_128_128 (verified reuse)
Experiment 42/142 campaign maximum | flat: custom_32_32_32_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_201708_659181_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 42/142 maximum | flat

Experiment 42/142 maximum | flat: custom_32_32_32_32 | params=262,379
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_201708_659181_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_201708_659181_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_201708_659181_abnormal-activities-dataset_capacity/models/custom_32_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.3441 selected_epoch=253 time=551s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9166666666666666, 'recall': 0.8461538461538461, 'f1': 0.88, 'support': 13, 'tp': 11, 'fp': 1, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_201708_659181_abnormal-activities-dataset_capacity/summary.json
Experiment 43/142 campaign maximum | flat: custom_64_64_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_202623_541051_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 43/142 maximum | flat

Experiment 43/142 maximum | flat: custom_64_64_64_64 | params=1,040,843
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_202623_541051_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_202623_541051_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_202623_541051_abnormal-activities-dataset_capacity/models/custom_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9563 val_loss=0.2286 selected_epoch=249 time=1075s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8, 'recall': 0.9230769230769231, 'f1': 0.8571428571428571, 'support': 13, 'tp': 12, 'fp': 3, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9166666666666666, 'recall': 0.7333333333333333, 'f1': 0.8148148148148148, 'support': 15, 'tp': 11, 'fp': 1, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_202623_541051_abnormal-activities-dataset_capacity/summary.json
Experiment 44/142 campaign maximum | flat: custom_96_96_96_96 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_204424_011632_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 44/142 maximum | flat

Experiment 44/142 maximum | flat: custom_96_96_96_96 | params=2,335,403
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_204424_011632_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_204424_011632_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_204424_011632_abnormal-activities-dataset_capacity/models/custom_96_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1849 selected_epoch=220 time=1932s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_204424_011632_abnormal-activities-dataset_capacity/summary.json
Experiment 45/142 campaign maximum | flat: custom_128_128_128_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_211642_554005_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 45/142 maximum | flat

Experiment 45/142 maximum | flat: custom_128_128_128_128 | params=4,146,059
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_211642_554005_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_211642_554005_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_211642_554005_abnormal-activities-dataset_capacity/models/custom_128_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9500 val_loss=0.4070 selected_epoch=159 time=1911s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.75, 'recall': 0.9230769230769231, 'f1': 0.8275862068965517, 'support': 13, 'tp': 12, 'fp': 4, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9166666666666666, 'recall': 0.7333333333333333, 'f1': 0.8148148148148148, 'support': 15, 'tp': 11, 'fp': 1, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_211642_554005_abnormal-activities-dataset_capacity/summary.json
Experiment 46/142 campaign maximum | flat: custom_8_8_8 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_214841_665123_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 46/142 maximum | flat

Experiment 46/142 maximum | flat: custom_8_8_8 | params=12,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_214841_665123_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_214841_665123_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_214841_665123_abnormal-activities-dataset_capacity/models/custom_8_8_8/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.4625 val_loss=1.3590 selected_epoch=254 time=248s
{'class_name': 'Begging', 'precision': 0.3953488372093023, 'recall': 0.85, 'f1': 0.5396825396825397, 'support': 20, 'tp': 17, 'fp': 26, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0.4, 'recall': 0.3333333333333333, 'f1': 0.36363636363636365, 'support': 12, 'tp': 4, 'fp': 6, 'fn': 8}
{'class_name': 'Knife Hazard', 'precision': 0.3888888888888889, 'recall': 0.4666666666666667, 'f1': 0.42424242424242425, 'support': 15, 'tp': 7, 'fp': 11, 'fn': 8}
{'class_name': 'Normal Videos', 'precision': 0.5769230769230769, 'recall': 0.8823529411764706, 'f1': 0.6976744186046512, 'support': 17, 'tp': 15, 'fp': 11, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.42857142857142855, 'recall': 0.8823529411764706, 'f1': 0.5769230769230769, 'support': 17, 'tp': 15, 'fp': 20, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 0.56, 'recall': 1.0, 'f1': 0.717948717948718, 'support': 14, 'tp': 14, 'fp': 11, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0.6666666666666666, 'recall': 0.18181818181818182, 'f1': 0.2857142857142857, 'support': 11, 'tp': 2, 'fp': 1, 'fn': 9}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_214841_665123_abnormal-activities-dataset_capacity/summary.json
Experiment 47/142 campaign maximum | flat: custom_16_16_16 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215253_522190_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 47/142 maximum | flat

Experiment 47/142 maximum | flat: custom_16_16_16 | params=48,187
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215253_522190_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215253_522190_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215253_522190_abnormal-activities-dataset_capacity/models/custom_16_16_16/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.7063 val_loss=0.9166 selected_epoch=256 time=368s
{'class_name': 'Begging', 'precision': 0.8636363636363636, 'recall': 0.95, 'f1': 0.9047619047619048, 'support': 20, 'tp': 19, 'fp': 3, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.5, 'recall': 0.23076923076923078, 'f1': 0.3157894736842105, 'support': 13, 'tp': 3, 'fp': 3, 'fn': 10}
{'class_name': 'Fight', 'precision': 0.4, 'recall': 0.14285714285714285, 'f1': 0.21052631578947367, 'support': 14, 'tp': 2, 'fp': 3, 'fn': 12}
{'class_name': 'Harassment', 'precision': 0.7272727272727273, 'recall': 0.5714285714285714, 'f1': 0.64, 'support': 14, 'tp': 8, 'fp': 3, 'fn': 6}
{'class_name': 'Hijack', 'precision': 0.6428571428571429, 'recall': 0.75, 'f1': 0.6923076923076923, 'support': 12, 'tp': 9, 'fp': 5, 'fn': 3}
{'class_name': 'Knife Hazard', 'precision': 0.42857142857142855, 'recall': 0.8, 'f1': 0.5581395348837209, 'support': 15, 'tp': 12, 'fp': 16, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.65, 'recall': 0.7647058823529411, 'f1': 0.7027027027027027, 'support': 17, 'tp': 13, 'fp': 7, 'fn': 4}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.6666666666666666, 'recall': 0.5454545454545454, 'f1': 0.6, 'support': 11, 'tp': 6, 'fp': 3, 'fn': 5}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215253_522190_abnormal-activities-dataset_capacity/summary.json
Experiment 48/142 campaign maximum | flat: custom_48_48_48 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215904_677181_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 48/142 maximum | flat

Experiment 48/142 maximum | flat: custom_48_48_48 | params=421,019
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215904_677181_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215904_677181_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215904_677181_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2975 selected_epoch=250 time=633s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_215904_677181_abnormal-activities-dataset_capacity/summary.json
Experiment 49/142 campaign maximum | flat: custom_80_80_80 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_220941_961403_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 49/142 maximum | flat

Experiment 49/142 maximum | flat: custom_80_80_80 | params=1,162,491
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_220941_961403_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_220941_961403_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_220941_961403_abnormal-activities-dataset_capacity/models/custom_80_80_80/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.3246 selected_epoch=242 time=1156s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_220941_961403_abnormal-activities-dataset_capacity/summary.json
Experiment 50/142 campaign maximum | flat: custom_64_64_64_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_222903_070956_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 50/142 maximum | flat

Experiment 50/142 maximum | flat: custom_64_64_64_64_64 | params=1,336,011
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_222903_070956_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_222903_070956_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_222903_070956_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=50 time=386s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_222903_070956_abnormal-activities-dataset_capacity/summary.json

Full results: 52 completed; 0 failed/pending
  1 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  2 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
  3 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
  4 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
  5 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
  6 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
  7 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
  8 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
  9 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 10 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 11 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 12 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 13 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 14 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 15 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 16 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 17 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 18 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 19 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 20 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 21 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 22 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 23 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 24 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 25 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 26 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 27 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 28 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 29 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 30 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 31 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 32 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 33 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 34 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 35 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 36 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 37 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 38 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 39 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
 40 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
 41 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
 42 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
 43 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
 44 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
 45 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
 46 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
 47 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
 48 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
 49 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 50 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
 51 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
 52 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): references, refinement, confirmation, weight_decay, temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (2,245,169,320 bytes; 1260 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 51/142 campaign maximum | references: r3d_18 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223806_601881_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 51/142 maximum | references

Experiment 51/142 maximum | references: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223806_601881_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223806_601881_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223806_601881_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8875 val_loss=0.3644 selected_epoch=36 time=60s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.7058823529411765, 'recall': 0.9230769230769231, 'f1': 0.8, 'support': 13, 'tp': 12, 'fp': 5, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.8666666666666667, 'recall': 0.9285714285714286, 'f1': 0.896551724137931, 'support': 14, 'tp': 13, 'fp': 2, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.7777777777777778, 'recall': 1.0, 'f1': 0.875, 'support': 14, 'tp': 14, 'fp': 4, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9, 'recall': 0.75, 'f1': 0.8181818181818182, 'support': 12, 'tp': 9, 'fp': 1, 'fn': 3}
{'class_name': 'Knife Hazard', 'precision': 0.9090909090909091, 'recall': 0.6666666666666666, 'f1': 0.7692307692307693, 'support': 15, 'tp': 10, 'fp': 1, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 0.8235294117647058, 'f1': 0.9032258064516129, 'support': 17, 'tp': 14, 'fp': 0, 'fn': 3}
{'class_name': 'Pollution', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Terrorism', 'precision': 0.7333333333333333, 'recall': 1.0, 'f1': 0.8461538461538461, 'support': 11, 'tp': 11, 'fp': 4, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223806_601881_abnormal-activities-dataset_capacity/summary.json
Experiment 52/142 campaign maximum | references: swin3d_t 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223916_447153_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 52/142 maximum | references

Experiment 52/142 maximum | references: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223916_447153_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223916_447153_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223916_447153_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=116 time=312s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_223916_447153_abnormal-activities-dataset_capacity/summary.json
Experiment 8/142 maximum | references r3d_18 (verified reuse)
Experiment 12/142 maximum | references swin3d_t (verified reuse)

Full results: 56 completed; 0 failed/pending
  1 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  2 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
  3 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
  4 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
  5 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
  6 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
  7 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
  8 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
  9 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 10 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 11 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 12 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 13 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 14 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 15 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 16 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 17 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 18 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 19 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 20 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 21 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 22 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 23 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 24 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 25 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 26 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 27 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 28 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 29 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 30 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 31 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 32 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 33 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 34 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 35 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 36 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 37 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 38 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 39 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 40 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 41 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
 42 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
 43 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
 44 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
 45 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
 46 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
 47 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
 48 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
 49 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
 50 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
 51 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 52 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 53 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 54 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
 55 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
 56 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): refinement, confirmation, weight_decay, temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (2,806,699,790 bytes; 1306 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 53/142 campaign maximum | refinement: custom_128_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_224754_944763_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 53/142 maximum | refinement

Experiment 53/142 maximum | refinement: custom_128_64_64 | params=1,342,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_224754_944763_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_224754_944763_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_224754_944763_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1490 selected_epoch=205 time=370s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_224754_944763_abnormal-activities-dataset_capacity/summary.json
Experiment 54/142 campaign maximum | refinement: custom_64_128_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225408_963685_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 54/142 maximum | refinement

Experiment 54/142 maximum | refinement: custom_64_128_64 | params=1,483,211
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225408_963685_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225408_963685_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225408_963685_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2155 selected_epoch=137 time=254s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225408_963685_abnormal-activities-dataset_capacity/summary.json
Experiment 55/142 campaign maximum | refinement: custom_64_64_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225827_320415_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 55/142 maximum | refinement

Experiment 55/142 maximum | refinement: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225827_320415_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225827_320415_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225827_320415_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1849 selected_epoch=162 time=275s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_225827_320415_abnormal-activities-dataset_capacity/summary.json
Experiment 56/142 campaign maximum | refinement: custom_32_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230305_984108_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 56/142 maximum | refinement

Experiment 56/142 maximum | refinement: custom_32_64_64 | params=557,771
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230305_984108_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230305_984108_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230305_984108_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1704 selected_epoch=174 time=217s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230305_984108_abnormal-activities-dataset_capacity/summary.json
Experiment 57/142 campaign maximum | refinement: custom_64_32_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230646_226930_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 57/142 maximum | refinement

Experiment 57/142 maximum | refinement: custom_64_32_64 | params=487,499
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230646_226930_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230646_226930_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230646_226930_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1752 selected_epoch=187 time=228s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_230646_226930_abnormal-activities-dataset_capacity/summary.json
Experiment 58/142 campaign maximum | refinement: custom_64_64_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231037_851089_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 58/142 maximum | refinement

Experiment 58/142 maximum | refinement: custom_64_64_32 | params=560,875
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231037_851089_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231037_851089_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231037_851089_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1557 selected_epoch=217 time=266s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231037_851089_abnormal-activities-dataset_capacity/summary.json
Experiment 59/142 campaign maximum | refinement: custom_96_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231507_281003_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 59/142 maximum | refinement

Experiment 59/142 maximum | refinement: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231507_281003_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231507_281003_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231507_281003_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1041 selected_epoch=175 time=281s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 12, 'tp': 12, 'fp': 0, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231507_281003_abnormal-activities-dataset_capacity/summary.json
Experiment 60/142 campaign maximum | refinement: custom_64_96_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231952_678729_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 60/142 maximum | refinement

Experiment 60/142 maximum | refinement: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231952_678729_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231952_678729_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231952_678729_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1794 selected_epoch=210 time=329s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 15, 'tp': 15, 'fp': 0, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_231952_678729_abnormal-activities-dataset_capacity/summary.json
Experiment 61/142 campaign maximum | refinement: custom_64_64_96 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_232525_398298_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 61/142 maximum | refinement

Experiment 61/142 maximum | refinement: custom_64_64_96 | params=1,004,203
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_232525_398298_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_232525_398298_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_232525_398298_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.1971 selected_epoch=195 time=287s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_232525_398298_abnormal-activities-dataset_capacity/summary.json
Experiment 62/142 campaign maximum | refinement: custom_32_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233016_454548_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 62/142 maximum | refinement

Experiment 62/142 maximum | refinement: custom_32_64 | params=262,603
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233016_454548_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233016_454548_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233016_454548_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.3373 selected_epoch=248 time=207s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233016_454548_abnormal-activities-dataset_capacity/summary.json
Experiment 63/142 campaign maximum | refinement: custom_64_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233347_385459_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 63/142 maximum | refinement

Experiment 63/142 maximum | refinement: custom_64_32 | params=265,707
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233347_385459_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233347_385459_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233347_385459_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2209 selected_epoch=255 time=209s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233347_385459_abnormal-activities-dataset_capacity/summary.json
Experiment 64/142 campaign maximum | refinement: custom_16_32_32 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233720_375657_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 64/142 maximum | refinement

Experiment 64/142 maximum | refinement: custom_16_32_32 | params=140,651
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233720_375657_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233720_375657_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233720_375657_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9438 val_loss=0.3268 selected_epoch=250 time=241s
{'class_name': 'Begging', 'precision': 0.9047619047619048, 'recall': 0.95, 'f1': 0.926829268292683, 'support': 20, 'tp': 19, 'fp': 2, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8461538461538461, 'recall': 0.8461538461538461, 'f1': 0.8461538461538461, 'support': 13, 'tp': 11, 'fp': 2, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9230769230769231, 'recall': 0.8, 'f1': 0.8571428571428571, 'support': 15, 'tp': 12, 'fp': 1, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_233720_375657_abnormal-activities-dataset_capacity/summary.json
Experiment 65/142 campaign maximum | refinement: custom_128_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_234125_232660_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 65/142 maximum | refinement

Experiment 65/142 maximum | refinement: custom_128_64_64 | params=1,342,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_234125_232660_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_234125_232660_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_234125_232660_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3176 selected_epoch=243 time=1276s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261010_234125_232660_abnormal-activities-dataset_capacity/summary.json
Experiment 66/142 campaign maximum | refinement: custom_64_128_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_000247_719673_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 66/142 maximum | refinement

Experiment 66/142 maximum | refinement: custom_64_128_64 | params=1,483,211
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_000247_719673_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_000247_719673_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_000247_719673_abnormal-activities-dataset_capacity/models/custom_64_128_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.2586 selected_epoch=203 time=1096s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_000247_719673_abnormal-activities-dataset_capacity/summary.json
Experiment 67/142 campaign maximum | refinement: custom_64_64_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_002108_984988_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 67/142 maximum | refinement

Experiment 67/142 maximum | refinement: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_002108_984988_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_002108_984988_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_002108_984988_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.2907 selected_epoch=172 time=870s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_002108_984988_abnormal-activities-dataset_capacity/summary.json
Experiment 68/142 campaign maximum | refinement: custom_32_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_003544_723581_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 68/142 maximum | refinement

Experiment 68/142 maximum | refinement: custom_32_64_64 | params=557,771
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_003544_723581_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_003544_723581_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_003544_723581_abnormal-activities-dataset_capacity/models/custom_32_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.4077 selected_epoch=256 time=686s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.8823529411764706, 'recall': 1.0, 'f1': 0.9375, 'support': 15, 'tp': 15, 'fp': 2, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_003544_723581_abnormal-activities-dataset_capacity/summary.json
Experiment 69/142 campaign maximum | refinement: custom_64_32_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_004715_562243_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 69/142 maximum | refinement

Experiment 69/142 maximum | refinement: custom_64_32_64 | params=487,499
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_004715_562243_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_004715_562243_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_004715_562243_abnormal-activities-dataset_capacity/models/custom_64_32_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.3853 selected_epoch=256 time=679s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_004715_562243_abnormal-activities-dataset_capacity/summary.json
Experiment 70/142 campaign maximum | refinement: custom_64_64_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_005839_372560_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 70/142 maximum | refinement

Experiment 70/142 maximum | refinement: custom_64_64_32 | params=560,875
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_005839_372560_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_005839_372560_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_005839_372560_abnormal-activities-dataset_capacity/models/custom_64_64_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2308 selected_epoch=237 time=707s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_005839_372560_abnormal-activities-dataset_capacity/summary.json
Experiment 71/142 campaign maximum | refinement: custom_96_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_011031_311303_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 71/142 maximum | refinement

Experiment 71/142 maximum | refinement: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_011031_311303_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_011031_311303_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_011031_311303_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9937 val_loss=0.2977 selected_epoch=235 time=1063s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 15, 'tp': 15, 'fp': 0, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_011031_311303_abnormal-activities-dataset_capacity/summary.json
Experiment 72/142 campaign maximum | refinement: custom_64_96_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_012819_679380_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 72/142 maximum | refinement

Experiment 72/142 maximum | refinement: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_012819_679380_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_012819_679380_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_012819_679380_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.2417 selected_epoch=238 time=1071s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_012819_679380_abnormal-activities-dataset_capacity/summary.json
Experiment 73/142 campaign maximum | refinement: custom_64_64_96 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_014615_917629_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 73/142 maximum | refinement

Experiment 73/142 maximum | refinement: custom_64_64_96 | params=1,004,203
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_014615_917629_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_014615_917629_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_014615_917629_abnormal-activities-dataset_capacity/models/custom_64_64_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2780 selected_epoch=222 time=943s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_014615_917629_abnormal-activities-dataset_capacity/summary.json
Experiment 74/142 campaign maximum | refinement: custom_32_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020204_449587_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 74/142 maximum | refinement

Experiment 74/142 maximum | refinement: custom_32_64 | params=262,603
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020204_449587_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020204_449587_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020204_449587_abnormal-activities-dataset_capacity/models/custom_32_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9250 val_loss=0.5206 selected_epoch=253 time=428s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8333333333333334, 'recall': 0.7692307692307693, 'f1': 0.8, 'support': 13, 'tp': 10, 'fp': 2, 'fn': 3}
{'class_name': 'Fight', 'precision': 0.8235294117647058, 'recall': 1.0, 'f1': 0.9032258064516129, 'support': 14, 'tp': 14, 'fp': 3, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.9230769230769231, 'recall': 0.8571428571428571, 'f1': 0.8888888888888888, 'support': 14, 'tp': 12, 'fp': 1, 'fn': 2}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9230769230769231, 'recall': 0.8, 'f1': 0.8571428571428571, 'support': 15, 'tp': 12, 'fp': 1, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.85, 'recall': 1.0, 'f1': 0.918918918918919, 'support': 17, 'tp': 17, 'fp': 3, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020204_449587_abnormal-activities-dataset_capacity/summary.json
Experiment 75/142 campaign maximum | refinement: custom_64_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020916_518676_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 75/142 maximum | refinement

Experiment 75/142 maximum | refinement: custom_64_32 | params=265,707
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020916_518676_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020916_518676_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020916_518676_abnormal-activities-dataset_capacity/models/custom_64_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9125 val_loss=0.5685 selected_epoch=256 time=444s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6875, 'recall': 0.8461538461538461, 'f1': 0.7586206896551724, 'support': 13, 'tp': 11, 'fp': 5, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.9090909090909091, 'recall': 0.7142857142857143, 'f1': 0.8, 'support': 14, 'tp': 10, 'fp': 1, 'fn': 4}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 0.9166666666666666, 'f1': 0.9565217391304348, 'support': 12, 'tp': 11, 'fp': 0, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.8, 'recall': 0.8, 'f1': 0.8, 'support': 15, 'tp': 12, 'fp': 3, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.8888888888888888, 'recall': 0.9411764705882353, 'f1': 0.9142857142857143, 'support': 17, 'tp': 16, 'fp': 2, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.8888888888888888, 'recall': 0.9411764705882353, 'f1': 0.9142857142857143, 'support': 17, 'tp': 16, 'fp': 2, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_020916_518676_abnormal-activities-dataset_capacity/summary.json
Experiment 76/142 campaign maximum | refinement: custom_16_32_32 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_021644_347633_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 76/142 maximum | refinement

Experiment 76/142 maximum | refinement: custom_16_32_32 | params=140,651
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_021644_347633_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_021644_347633_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_021644_347633_abnormal-activities-dataset_capacity/models/custom_16_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8375 val_loss=0.6566 selected_epoch=255 time=373s
{'class_name': 'Begging', 'precision': 0.9, 'recall': 0.9, 'f1': 0.9, 'support': 20, 'tp': 18, 'fp': 2, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.5714285714285714, 'recall': 0.3076923076923077, 'f1': 0.4, 'support': 13, 'tp': 4, 'fp': 3, 'fn': 9}
{'class_name': 'Fight', 'precision': 0.8461538461538461, 'recall': 0.7857142857142857, 'f1': 0.8148148148148148, 'support': 14, 'tp': 11, 'fp': 2, 'fn': 3}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.7857142857142857, 'f1': 0.88, 'support': 14, 'tp': 11, 'fp': 0, 'fn': 3}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 0.8333333333333334, 'f1': 0.9090909090909091, 'support': 12, 'tp': 10, 'fp': 0, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.5714285714285714, 'recall': 0.8, 'f1': 0.6666666666666666, 'support': 15, 'tp': 12, 'fp': 9, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.8333333333333334, 'recall': 0.8823529411764706, 'f1': 0.8571428571428571, 'support': 17, 'tp': 15, 'fp': 3, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.75, 'recall': 0.8823529411764706, 'f1': 0.8108108108108109, 'support': 17, 'tp': 15, 'fp': 5, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_021644_347633_abnormal-activities-dataset_capacity/summary.json

Full results: 80 completed; 0 failed/pending
  1 refinement custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.99375 val_loss=0.29771 params=1007307
  2 refinement custom_32_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.170426 params=557771
  3 refinement custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.179411 params=1077579
  4 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  5 refinement custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.241714 params=1077579
  6 refinement custom_64_128_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.258579 params=1483211
  7 refinement custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.290738 params=1336459
  8 refinement custom_128_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.149044 params=1342667
  9 refinement custom_64_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.155748 params=560875
 10 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
 11 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
 12 refinement custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.184906 params=1336459
 13 refinement custom_64_64_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.197078 params=1004203
 14 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
 15 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
 16 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
 17 refinement custom_128_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.317599 params=1342667
 18 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 19 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 20 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 21 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 22 refinement custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.104137 params=1007307
 23 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 24 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 25 refinement custom_64_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.175208 params=487499
 26 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 27 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 28 refinement custom_64_128_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.215485 params=1483211
 29 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 30 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 31 refinement custom_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.220924 params=265707
 32 refinement custom_64_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.230771 params=560875
 33 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 34 refinement custom_64_64_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.277994 params=1004203
 35 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 36 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 37 refinement custom_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.337258 params=262603
 38 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 39 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 40 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 41 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 42 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 43 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 44 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 45 refinement custom_64_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.385308 params=487499
 46 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 47 refinement custom_32_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.407691 params=557771
 48 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 49 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 50 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 51 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 52 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 53 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 54 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 55 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 56 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 57 refinement custom_16_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.3268 params=140651
 58 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 59 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 60 refinement custom_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.925 val_loss=0.520644 params=262603
 61 refinement custom_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9125 val_loss=0.568474 params=265707
 62 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 63 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 64 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
 65 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
 66 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
 67 refinement custom_16_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8375 val_loss=0.656602 params=140651
 68 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
 69 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
 70 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
 71 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
 72 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
 73 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
 74 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
 75 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 76 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 77 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 78 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
 79 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
 80 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): confirmation, weight_decay, temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (3,025,655,627 bytes; 1906 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 77/142 campaign maximum | confirmation: custom_64_96_64 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_022625_983546_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 77/142 maximum | confirmation

Experiment 77/142 maximum | confirmation: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_022625_983546_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_022625_983546_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_022625_983546_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1935 selected_epoch=162 time=259s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_022625_983546_abnormal-activities-dataset_capacity/summary.json
Experiment 78/142 campaign maximum | confirmation: custom_96_64_64 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023049_461540_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 78/142 maximum | confirmation

Experiment 78/142 maximum | confirmation: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023049_461540_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023049_461540_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023049_461540_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1448 selected_epoch=207 time=319s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023049_461540_abnormal-activities-dataset_capacity/summary.json
Experiment 79/142 campaign maximum | confirmation: custom_64_64_128 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023612_538460_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 79/142 maximum | confirmation

Experiment 79/142 maximum | confirmation: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023612_538460_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023612_538460_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023612_538460_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2919 selected_epoch=147 time=250s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_023612_538460_abnormal-activities-dataset_capacity/summary.json
Experiment 80/142 campaign maximum | confirmation: custom_48_48_48 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024026_282465_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 80/142 maximum | confirmation

Experiment 80/142 maximum | confirmation: custom_48_48_48 | params=421,019
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024026_282465_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024026_282465_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024026_282465_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2195 selected_epoch=242 time=260s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024026_282465_abnormal-activities-dataset_capacity/summary.json
Experiment 81/142 campaign maximum | confirmation: custom_128_64_64 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024450_259627_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 81/142 maximum | confirmation

Experiment 81/142 maximum | confirmation: custom_128_64_64 | params=1,342,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024450_259627_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024450_259627_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024450_259627_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.0980 selected_epoch=167 time=304s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024450_259627_abnormal-activities-dataset_capacity/summary.json
Experiment 82/142 campaign maximum | confirmation: custom_32_32_32 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024958_488214_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 82/142 maximum | confirmation

Experiment 82/142 maximum | confirmation: custom_32_32_32 | params=188,523
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024958_488214_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024958_488214_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024958_488214_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.2914 selected_epoch=254 time=231s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8666666666666667, 'recall': 1.0, 'f1': 0.9285714285714286, 'support': 13, 'tp': 13, 'fp': 2, 'fn': 0}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.8461538461538461, 'recall': 0.9166666666666666, 'f1': 0.88, 'support': 12, 'tp': 11, 'fp': 2, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 1.0, 'recall': 0.8666666666666667, 'f1': 0.9285714285714286, 'support': 15, 'tp': 13, 'fp': 0, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 11, 'tp': 10, 'fp': 1, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_024958_488214_abnormal-activities-dataset_capacity/summary.json
Experiment 83/142 campaign maximum | confirmation: custom_64_64_64 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025353_215866_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 83/142 maximum | confirmation

Experiment 83/142 maximum | confirmation: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025353_215866_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025353_215866_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025353_215866_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2055 selected_epoch=151 time=205s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025353_215866_abnormal-activities-dataset_capacity/summary.json
Experiment 84/142 campaign maximum | confirmation: custom_96_96_96 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025721_857767_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 84/142 maximum | confirmation

Experiment 84/142 maximum | confirmation: custom_96_96_96 | params=1,671,467
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025721_857767_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025721_857767_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025721_857767_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2581 selected_epoch=125 time=259s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 12, 'tp': 12, 'fp': 0, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_025721_857767_abnormal-activities-dataset_capacity/summary.json
Experiment 85/142 campaign maximum | confirmation: custom_128_128_128 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030145_057391_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 85/142 maximum | confirmation

Experiment 85/142 maximum | confirmation: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030145_057391_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030145_057391_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030145_057391_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.1814 selected_epoch=100 time=288s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 12, 'tp': 12, 'fp': 0, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030145_057391_abnormal-activities-dataset_capacity/summary.json
Experiment 86/142 campaign maximum | confirmation: custom_64_64_64_64_64 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030637_962682_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 86/142 maximum | confirmation

Experiment 86/142 maximum | confirmation: custom_64_64_64_64_64 | params=1,336,011
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030637_962682_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030637_962682_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030637_962682_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9875 val_loss=0.1749 selected_epoch=194 time=374s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_030637_962682_abnormal-activities-dataset_capacity/summary.json
Experiment 87/142 campaign maximum | confirmation: r3d_18 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031256_698148_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 87/142 maximum | confirmation

Experiment 87/142 maximum | confirmation: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031256_698148_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031256_698148_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031256_698148_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8875 val_loss=0.4670 selected_epoch=26 time=57s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6666666666666666, 'recall': 0.7692307692307693, 'f1': 0.7142857142857143, 'support': 13, 'tp': 10, 'fp': 5, 'fn': 3}
{'class_name': 'Fight', 'precision': 0.9230769230769231, 'recall': 0.8571428571428571, 'f1': 0.8888888888888888, 'support': 14, 'tp': 12, 'fp': 1, 'fn': 2}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.8571428571428571, 'f1': 0.9230769230769231, 'support': 14, 'tp': 12, 'fp': 0, 'fn': 2}
{'class_name': 'Hijack', 'precision': 0.9, 'recall': 0.75, 'f1': 0.8181818181818182, 'support': 12, 'tp': 9, 'fp': 1, 'fn': 3}
{'class_name': 'Knife Hazard', 'precision': 0.6923076923076923, 'recall': 0.6, 'f1': 0.6428571428571429, 'support': 15, 'tp': 9, 'fp': 4, 'fn': 6}
{'class_name': 'Normal Videos', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.7857142857142857, 'recall': 1.0, 'f1': 0.88, 'support': 11, 'tp': 11, 'fp': 3, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031256_698148_abnormal-activities-dataset_capacity/summary.json
Experiment 88/142 campaign maximum | confirmation: swin3d_t 32px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031403_000755_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 88/142 maximum | confirmation

Experiment 88/142 maximum | confirmation: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031403_000755_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031403_000755_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031403_000755_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=67 time=204s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031403_000755_abnormal-activities-dataset_capacity/summary.json
Experiment 89/142 campaign maximum | confirmation: custom_64_96_64 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031736_744964_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 89/142 maximum | confirmation

Experiment 89/142 maximum | confirmation: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031736_744964_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031736_744964_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031736_744964_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2899 selected_epoch=229 time=1056s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_031736_744964_abnormal-activities-dataset_capacity/summary.json
Experiment 90/142 campaign maximum | confirmation: custom_96_64_64 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_033519_676421_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 90/142 maximum | confirmation

Experiment 90/142 maximum | confirmation: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_033519_676421_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_033519_676421_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_033519_676421_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2295 selected_epoch=256 time=1063s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_033519_676421_abnormal-activities-dataset_capacity/summary.json
Experiment 91/142 campaign maximum | confirmation: custom_64_64_128 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_035307_699130_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 91/142 maximum | confirmation

Experiment 91/142 maximum | confirmation: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_035307_699130_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_035307_699130_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_035307_699130_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3604 selected_epoch=244 time=1142s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_035307_699130_abnormal-activities-dataset_capacity/summary.json
Experiment 92/142 campaign maximum | confirmation: custom_48_48_48 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_041215_056679_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 92/142 maximum | confirmation

Experiment 92/142 maximum | confirmation: custom_48_48_48 | params=421,019
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_041215_056679_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_041215_056679_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_041215_056679_abnormal-activities-dataset_capacity/models/custom_48_48_48/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.3771 selected_epoch=250 time=633s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8571428571428571, 'recall': 0.9230769230769231, 'f1': 0.8888888888888888, 'support': 13, 'tp': 12, 'fp': 2, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9230769230769231, 'recall': 0.8, 'f1': 0.8571428571428571, 'support': 15, 'tp': 12, 'fp': 1, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_041215_056679_abnormal-activities-dataset_capacity/summary.json
Experiment 93/142 campaign maximum | confirmation: custom_128_64_64 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_042252_250270_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 93/142 maximum | confirmation

Experiment 93/142 maximum | confirmation: custom_128_64_64 | params=1,342,667
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_042252_250270_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_042252_250270_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_042252_250270_abnormal-activities-dataset_capacity/models/custom_128_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1835 selected_epoch=208 time=1150s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_042252_250270_abnormal-activities-dataset_capacity/summary.json
Experiment 94/142 campaign maximum | confirmation: custom_32_32_32 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044208_797382_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 94/142 maximum | confirmation

Experiment 94/142 maximum | confirmation: custom_32_32_32 | params=188,523
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044208_797382_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044208_797382_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044208_797382_abnormal-activities-dataset_capacity/models/custom_32_32_32/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8438 val_loss=0.6406 selected_epoch=256 time=431s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.6, 'recall': 0.6923076923076923, 'f1': 0.6428571428571429, 'support': 13, 'tp': 9, 'fp': 6, 'fn': 4}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.7857142857142857, 'f1': 0.88, 'support': 14, 'tp': 11, 'fp': 0, 'fn': 3}
{'class_name': 'Harassment', 'precision': 0.9090909090909091, 'recall': 0.7142857142857143, 'f1': 0.8, 'support': 14, 'tp': 10, 'fp': 1, 'fn': 4}
{'class_name': 'Hijack', 'precision': 0.875, 'recall': 0.5833333333333334, 'f1': 0.7, 'support': 12, 'tp': 7, 'fp': 1, 'fn': 5}
{'class_name': 'Knife Hazard', 'precision': 0.6, 'recall': 0.6, 'f1': 0.6, 'support': 15, 'tp': 9, 'fp': 6, 'fn': 6}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.8125, 'recall': 1.0, 'f1': 0.896551724137931, 'support': 13, 'tp': 13, 'fp': 3, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.6875, 'recall': 1.0, 'f1': 0.8148148148148148, 'support': 11, 'tp': 11, 'fp': 5, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044208_797382_abnormal-activities-dataset_capacity/summary.json
Experiment 95/142 campaign maximum | confirmation: custom_64_64_64 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044923_597144_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 95/142 maximum | confirmation

Experiment 95/142 maximum | confirmation: custom_64_64_64 | params=745,675
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044923_597144_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044923_597144_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044923_597144_abnormal-activities-dataset_capacity/models/custom_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.3166 selected_epoch=236 time=810s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_044923_597144_abnormal-activities-dataset_capacity/summary.json
Experiment 96/142 campaign maximum | confirmation: custom_96_96_96 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_050258_758707_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 96/142 maximum | confirmation

Experiment 96/142 maximum | confirmation: custom_96_96_96 | params=1,671,467
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_050258_758707_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_050258_758707_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_050258_758707_abnormal-activities-dataset_capacity/models/custom_96_96_96/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.3135 selected_epoch=195 time=1284s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_050258_758707_abnormal-activities-dataset_capacity/summary.json
Experiment 97/142 campaign maximum | confirmation: custom_128_128_128 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_052428_494758_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 97/142 maximum | confirmation

Experiment 97/142 maximum | confirmation: custom_128_128_128 | params=2,965,899
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_052428_494758_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_052428_494758_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_052428_494758_abnormal-activities-dataset_capacity/models/custom_128_128_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2752 selected_epoch=170 time=1508s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_052428_494758_abnormal-activities-dataset_capacity/summary.json
Experiment 98/142 campaign maximum | confirmation: custom_64_64_64_64_64 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_054943_609093_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 98/142 maximum | confirmation

Experiment 98/142 maximum | confirmation: custom_64_64_64_64_64 | params=1,336,011
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_054943_609093_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_054943_609093_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_054943_609093_abnormal-activities-dataset_capacity/models/custom_64_64_64_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3843 selected_epoch=23 time=332s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_054943_609093_abnormal-activities-dataset_capacity/summary.json
Experiment 99/142 campaign maximum | confirmation: r3d_18 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055521_164819_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 99/142 maximum | confirmation

Experiment 99/142 maximum | confirmation: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055521_164819_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055521_164819_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055521_164819_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9062 val_loss=0.3690 selected_epoch=31 time=81s
{'class_name': 'Begging', 'precision': 0.9047619047619048, 'recall': 0.95, 'f1': 0.926829268292683, 'support': 20, 'tp': 19, 'fp': 2, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8461538461538461, 'recall': 0.8461538461538461, 'f1': 0.8461538461538461, 'support': 13, 'tp': 11, 'fp': 2, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.9230769230769231, 'recall': 0.8571428571428571, 'f1': 0.8888888888888888, 'support': 14, 'tp': 12, 'fp': 1, 'fn': 2}
{'class_name': 'Harassment', 'precision': 0.9230769230769231, 'recall': 0.8571428571428571, 'f1': 0.8888888888888888, 'support': 14, 'tp': 12, 'fp': 1, 'fn': 2}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.7857142857142857, 'recall': 0.7333333333333333, 'f1': 0.7586206896551724, 'support': 15, 'tp': 11, 'fp': 3, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.8888888888888888, 'recall': 0.9411764705882353, 'f1': 0.9142857142857143, 'support': 17, 'tp': 16, 'fp': 2, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.8888888888888888, 'recall': 0.9411764705882353, 'f1': 0.9142857142857143, 'support': 17, 'tp': 16, 'fp': 2, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055521_164819_abnormal-activities-dataset_capacity/summary.json
Experiment 100/142 campaign maximum | confirmation: swin3d_t 64px, 8f/16fps, seed 2026
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055652_485861_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 100/142 maximum | confirmation

Experiment 100/142 maximum | confirmation: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055652_485861_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055652_485861_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055652_485861_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=83 time=265s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_055652_485861_abnormal-activities-dataset_capacity/summary.json

Full results: 104 completed; 0 failed/pending
  1 refinement custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.99375 val_loss=0.29771 params=1007307
  2 confirmation custom_128_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.0979638 params=1342667
  3 confirmation custom_96_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.144812 params=1007307
  4 refinement custom_32_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.170426 params=557771
  5 confirmation custom_64_64_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.174946 params=1336011
  6 refinement custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.179411 params=1077579
  7 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  8 confirmation custom_64_96_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.193503 params=1077579
  9 refinement custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.241714 params=1077579
 10 refinement custom_64_128_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.258579 params=1483211
 11 refinement custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.290738 params=1336459
 12 refinement custom_128_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.149044 params=1342667
 13 refinement custom_64_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.155748 params=560875
 14 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
 15 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
 16 refinement custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.184906 params=1336459
 17 refinement custom_64_64_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.197078 params=1004203
 18 confirmation custom_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.205548 params=745675
 19 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
 20 confirmation custom_96_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.229499 params=1007307
 21 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
 22 confirmation custom_64_96_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.289887 params=1077579
 23 confirmation custom_64_64_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.291864 params=1336459
 24 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
 25 confirmation custom_96_96_96 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.313502 params=1671467
 26 refinement custom_128_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.317599 params=1342667
 27 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 28 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 29 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 30 confirmation custom_64_64_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.360393 params=1336459
 31 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 32 refinement custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.104137 params=1007307
 33 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 34 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 35 refinement custom_64_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.175208 params=487499
 36 confirmation custom_128_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.183511 params=1342667
 37 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 38 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 39 refinement custom_64_128_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.215485 params=1483211
 40 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 41 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 42 confirmation custom_48_48_48 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.219522 params=421019
 43 refinement custom_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.220924 params=265707
 44 refinement custom_64_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.230771 params=560875
 45 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 46 confirmation custom_96_96_96 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.258062 params=1671467
 47 confirmation custom_128_128_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.275204 params=2965899
 48 refinement custom_64_64_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.277994 params=1004203
 49 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 50 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 51 refinement custom_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.337258 params=262603
 52 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 53 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 54 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 55 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 56 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 57 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 58 confirmation custom_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.316626 params=745675
 59 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 60 refinement custom_64_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.385308 params=487499
 61 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 62 refinement custom_32_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.407691 params=557771
 63 confirmation custom_128_128_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.181375 params=2965899
 64 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 65 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 66 confirmation custom_32_32_32 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.291422 params=188523
 67 confirmation custom_48_48_48 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.377144 params=421019
 68 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 69 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 70 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 71 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 72 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 73 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 74 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 75 refinement custom_16_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.3268 params=140651
 76 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 77 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 78 refinement custom_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.925 val_loss=0.520644 params=262603
 79 refinement custom_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9125 val_loss=0.568474 params=265707
 80 confirmation r3d_18 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.368961 params=33171915
 81 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 82 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 83 confirmation r3d_18 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.467042 params=33171915
 84 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
 85 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
 86 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
 87 confirmation custom_32_32_32 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.640571 params=188523
 88 refinement custom_16_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8375 val_loss=0.656602 params=140651
 89 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
 90 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
 91 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
 92 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
 93 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
 94 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
 95 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
 96 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 97 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 98 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
 99 confirmation swin3d_t complete | 64px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
100 confirmation swin3d_t complete | 32px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
101 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
102 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
103 confirmation custom_64_64_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.125 val_loss=2.3843 params=1336011
104 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): weight_decay, temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (4,425,373,998 bytes; 2498 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 101/142 campaign maximum | weight_decay: custom_64_96_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_060626_173872_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 101/142 maximum | weight_decay

Experiment 101/142 maximum | weight_decay: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_060626_173872_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_060626_173872_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_060626_173872_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1891 selected_epoch=246 time=355s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_060626_173872_abnormal-activities-dataset_capacity/summary.json
Experiment 102/142 campaign maximum | weight_decay: custom_64_96_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061225_887810_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 102/142 maximum | weight_decay

Experiment 102/142 maximum | weight_decay: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061225_887810_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061225_887810_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061225_887810_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9563 val_loss=0.2292 selected_epoch=227 time=350s
{'class_name': 'Begging', 'precision': 0.9473684210526315, 'recall': 0.9, 'f1': 0.9230769230769231, 'support': 20, 'tp': 18, 'fp': 1, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8461538461538461, 'recall': 0.9166666666666666, 'f1': 0.88, 'support': 12, 'tp': 11, 'fp': 2, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 11, 'tp': 10, 'fp': 1, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061225_887810_abnormal-activities-dataset_capacity/summary.json
Experiment 103/142 campaign maximum | weight_decay: custom_96_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061820_577482_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 103/142 maximum | weight_decay

Experiment 103/142 maximum | weight_decay: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061820_577482_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061820_577482_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061820_577482_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.1802 selected_epoch=133 time=217s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.875, 'recall': 1.0, 'f1': 0.9333333333333333, 'support': 14, 'tp': 14, 'fp': 2, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_061820_577482_abnormal-activities-dataset_capacity/summary.json
Experiment 104/142 campaign maximum | weight_decay: custom_96_64_64 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062202_089811_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 104/142 maximum | weight_decay

Experiment 104/142 maximum | weight_decay: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062202_089811_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062202_089811_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062202_089811_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9375 val_loss=0.2797 selected_epoch=149 time=239s
{'class_name': 'Begging', 'precision': 0.9047619047619048, 'recall': 0.95, 'f1': 0.926829268292683, 'support': 20, 'tp': 19, 'fp': 2, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.7857142857142857, 'f1': 0.88, 'support': 14, 'tp': 11, 'fp': 0, 'fn': 3}
{'class_name': 'Harassment', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9090909090909091, 'recall': 0.8333333333333334, 'f1': 0.8695652173913043, 'support': 12, 'tp': 10, 'fp': 1, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.8333333333333334, 'recall': 1.0, 'f1': 0.9090909090909091, 'support': 15, 'tp': 15, 'fp': 3, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8823529411764706, 'f1': 0.9375, 'support': 17, 'tp': 15, 'fp': 0, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062202_089811_abnormal-activities-dataset_capacity/summary.json
Experiment 105/142 campaign maximum | weight_decay: custom_64_64_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062605_757974_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 105/142 maximum | weight_decay

Experiment 105/142 maximum | weight_decay: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062605_757974_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062605_757974_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062605_757974_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9812 val_loss=0.2046 selected_epoch=154 time=260s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 12, 'tp': 12, 'fp': 2, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9375, 'recall': 1.0, 'f1': 0.967741935483871, 'support': 15, 'tp': 15, 'fp': 1, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'support': 11, 'tp': 10, 'fp': 0, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_062605_757974_abnormal-activities-dataset_capacity/summary.json
Experiment 106/142 campaign maximum | weight_decay: custom_64_64_128 32px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063030_012616_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 106/142 maximum | weight_decay

Experiment 106/142 maximum | weight_decay: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063030_012616_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063030_012616_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063030_012616_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.2276 selected_epoch=225 time=363s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.9, 'f1': 0.9473684210526315, 'support': 20, 'tp': 18, 'fp': 0, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063030_012616_abnormal-activities-dataset_capacity/summary.json
Experiment 107/142 campaign maximum | weight_decay: custom_64_96_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063637_094770_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 107/142 maximum | weight_decay

Experiment 107/142 maximum | weight_decay: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063637_094770_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063637_094770_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063637_094770_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2917 selected_epoch=251 time=1068s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8571428571428571, 'recall': 0.9230769230769231, 'f1': 0.8888888888888888, 'support': 13, 'tp': 12, 'fp': 2, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_063637_094770_abnormal-activities-dataset_capacity/summary.json
Experiment 108/142 campaign maximum | weight_decay: custom_64_96_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_065431_752819_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 108/142 maximum | weight_decay

Experiment 108/142 maximum | weight_decay: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_065431_752819_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_065431_752819_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_065431_752819_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8000 val_loss=0.7810 selected_epoch=252 time=1067s
{'class_name': 'Begging', 'precision': 0.8947368421052632, 'recall': 0.85, 'f1': 0.8717948717948718, 'support': 20, 'tp': 17, 'fp': 2, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0.42857142857142855, 'recall': 0.9230769230769231, 'f1': 0.5853658536585366, 'support': 13, 'tp': 12, 'fp': 16, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.8333333333333334, 'recall': 0.35714285714285715, 'f1': 0.5, 'support': 14, 'tp': 5, 'fp': 1, 'fn': 9}
{'class_name': 'Harassment', 'precision': 0.8125, 'recall': 0.9285714285714286, 'f1': 0.8666666666666667, 'support': 14, 'tp': 13, 'fp': 3, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9090909090909091, 'recall': 0.8333333333333334, 'f1': 0.8695652173913043, 'support': 12, 'tp': 10, 'fp': 1, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.75, 'recall': 0.4, 'f1': 0.5217391304347826, 'support': 15, 'tp': 6, 'fp': 2, 'fn': 9}
{'class_name': 'Normal Videos', 'precision': 0.8, 'recall': 0.9411764705882353, 'f1': 0.8648648648648649, 'support': 17, 'tp': 16, 'fp': 4, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.9230769230769231, 'recall': 0.7058823529411765, 'f1': 0.8, 'support': 17, 'tp': 12, 'fp': 1, 'fn': 5}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_065431_752819_abnormal-activities-dataset_capacity/summary.json
Experiment 109/142 campaign maximum | weight_decay: custom_96_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_071224_159243_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 109/142 maximum | weight_decay

Experiment 109/142 maximum | weight_decay: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_071224_159243_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_071224_159243_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_071224_159243_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9750 val_loss=0.2417 selected_epoch=243 time=1059s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_071224_159243_abnormal-activities-dataset_capacity/summary.json
Experiment 110/142 campaign maximum | weight_decay: custom_96_64_64 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_073008_478667_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 110/142 maximum | weight_decay

Experiment 110/142 maximum | weight_decay: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_073008_478667_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_073008_478667_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_073008_478667_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8500 val_loss=0.5930 selected_epoch=255 time=1059s
{'class_name': 'Begging', 'precision': 0.9047619047619048, 'recall': 0.95, 'f1': 0.926829268292683, 'support': 20, 'tp': 19, 'fp': 2, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.75, 'recall': 0.6923076923076923, 'f1': 0.72, 'support': 13, 'tp': 9, 'fp': 3, 'fn': 4}
{'class_name': 'Fight', 'precision': 0.8571428571428571, 'recall': 0.8571428571428571, 'f1': 0.8571428571428571, 'support': 14, 'tp': 12, 'fp': 2, 'fn': 2}
{'class_name': 'Harassment', 'precision': 0.7857142857142857, 'recall': 0.7857142857142857, 'f1': 0.7857142857142857, 'support': 14, 'tp': 11, 'fp': 3, 'fn': 3}
{'class_name': 'Hijack', 'precision': 0.9090909090909091, 'recall': 0.8333333333333334, 'f1': 0.8695652173913043, 'support': 12, 'tp': 10, 'fp': 1, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.7142857142857143, 'recall': 0.6666666666666666, 'f1': 0.6896551724137931, 'support': 15, 'tp': 10, 'fp': 4, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.8, 'recall': 0.9411764705882353, 'f1': 0.8648648648648649, 'support': 17, 'tp': 16, 'fp': 4, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.8235294117647058, 'recall': 0.8235294117647058, 'f1': 0.8235294117647058, 'support': 17, 'tp': 14, 'fp': 3, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.7692307692307693, 'f1': 0.8695652173913043, 'support': 13, 'tp': 10, 'fp': 0, 'fn': 3}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_073008_478667_abnormal-activities-dataset_capacity/summary.json
Experiment 111/142 campaign maximum | weight_decay: custom_64_64_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_074753_443504_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 111/142 maximum | weight_decay

Experiment 111/142 maximum | weight_decay: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_074753_443504_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_074753_443504_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_074753_443504_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9688 val_loss=0.2709 selected_epoch=250 time=1136s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.8823529411764706, 'recall': 1.0, 'f1': 0.9375, 'support': 15, 'tp': 15, 'fp': 2, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_074753_443504_abnormal-activities-dataset_capacity/summary.json
Experiment 112/142 campaign maximum | weight_decay: custom_64_64_128 64px, 8f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_080655_357820_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 112/142 maximum | weight_decay

Experiment 112/142 maximum | weight_decay: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_080655_357820_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_080655_357820_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_080655_357820_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.7875 val_loss=0.8289 selected_epoch=256 time=1138s
{'class_name': 'Begging', 'precision': 0.6785714285714286, 'recall': 0.95, 'f1': 0.7916666666666666, 'support': 20, 'tp': 19, 'fp': 9, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.46153846153846156, 'f1': 0.631578947368421, 'support': 13, 'tp': 6, 'fp': 0, 'fn': 7}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.35714285714285715, 'f1': 0.5263157894736842, 'support': 14, 'tp': 5, 'fp': 0, 'fn': 9}
{'class_name': 'Harassment', 'precision': 0.9, 'recall': 0.6428571428571429, 'f1': 0.75, 'support': 14, 'tp': 9, 'fp': 1, 'fn': 5}
{'class_name': 'Hijack', 'precision': 0.8888888888888888, 'recall': 0.6666666666666666, 'f1': 0.7619047619047619, 'support': 12, 'tp': 8, 'fp': 1, 'fn': 4}
{'class_name': 'Knife Hazard', 'precision': 0.5652173913043478, 'recall': 0.8666666666666667, 'f1': 0.6842105263157895, 'support': 15, 'tp': 13, 'fp': 10, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.875, 'recall': 0.8235294117647058, 'f1': 0.8484848484848485, 'support': 17, 'tp': 14, 'fp': 2, 'fn': 3}
{'class_name': 'Pollution', 'precision': 0.7142857142857143, 'recall': 0.8823529411764706, 'f1': 0.7894736842105263, 'support': 17, 'tp': 15, 'fp': 6, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9230769230769231, 'recall': 0.9230769230769231, 'f1': 0.9230769230769231, 'support': 13, 'tp': 12, 'fp': 1, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 0.7857142857142857, 'recall': 1.0, 'f1': 0.88, 'support': 11, 'tp': 11, 'fp': 3, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_080655_357820_abnormal-activities-dataset_capacity/summary.json

Full results: 116 completed; 0 failed/pending
  1 refinement custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.99375 val_loss=0.29771 params=1007307
  2 confirmation custom_128_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.0979638 params=1342667
  3 confirmation custom_96_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.144812 params=1007307
  4 refinement custom_32_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.170426 params=557771
  5 confirmation custom_64_64_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.174946 params=1336011
  6 refinement custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.179411 params=1077579
  7 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  8 confirmation custom_64_96_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.193503 params=1077579
  9 refinement custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.241714 params=1077579
 10 refinement custom_64_128_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.258579 params=1483211
 11 refinement custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.290738 params=1336459
 12 refinement custom_128_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.149044 params=1342667
 13 refinement custom_64_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.155748 params=560875
 14 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
 15 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
 16 refinement custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.184906 params=1336459
 17 refinement custom_64_64_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.197078 params=1004203
 18 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.98125 val_loss=0.204597 params=1336459
 19 confirmation custom_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.205548 params=745675
 20 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
 21 confirmation custom_96_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.229499 params=1007307
 22 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
 23 confirmation custom_64_96_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.289887 params=1077579
 24 confirmation custom_64_64_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.291864 params=1336459
 25 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
 26 confirmation custom_96_96_96 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.313502 params=1671467
 27 refinement custom_128_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.317599 params=1342667
 28 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 29 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 30 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 31 confirmation custom_64_64_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.360393 params=1336459
 32 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 33 refinement custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.104137 params=1007307
 34 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 35 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 36 refinement custom_64_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.175208 params=487499
 37 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.180159 params=1007307
 38 confirmation custom_128_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.183511 params=1342667
 39 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 40 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.189062 params=1077579
 41 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 42 refinement custom_64_128_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.215485 params=1483211
 43 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 44 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 45 confirmation custom_48_48_48 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.219522 params=421019
 46 refinement custom_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.220924 params=265707
 47 refinement custom_64_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.230771 params=560875
 48 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 49 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.241667 params=1007307
 50 confirmation custom_96_96_96 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.258062 params=1671467
 51 confirmation custom_128_128_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.275204 params=2965899
 52 refinement custom_64_64_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.277994 params=1004203
 53 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.291667 params=1077579
 54 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 55 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 56 refinement custom_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.337258 params=262603
 57 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 58 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 59 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 60 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 61 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 62 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.96875 val_loss=0.227579 params=1336459
 63 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 64 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.96875 val_loss=0.270876 params=1336459
 65 confirmation custom_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.316626 params=745675
 66 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 67 refinement custom_64_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.385308 params=487499
 68 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 69 refinement custom_32_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.407691 params=557771
 70 confirmation custom_128_128_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.181375 params=2965899
 71 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 72 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 73 confirmation custom_32_32_32 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.291422 params=188523
 74 confirmation custom_48_48_48 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.377144 params=421019
 75 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 76 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 77 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 78 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.95625 val_loss=0.229204 params=1077579
 79 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 80 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 81 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 82 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 83 refinement custom_16_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.3268 params=140651
 84 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 85 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.9375 val_loss=0.279749 params=1007307
 86 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 87 refinement custom_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.925 val_loss=0.520644 params=262603
 88 refinement custom_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9125 val_loss=0.568474 params=265707
 89 confirmation r3d_18 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.368961 params=33171915
 90 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 91 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 92 confirmation r3d_18 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.467042 params=33171915
 93 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
 94 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.85 val_loss=0.592975 params=1007307
 95 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
 96 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
 97 confirmation custom_32_32_32 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.640571 params=188523
 98 refinement custom_16_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8375 val_loss=0.656602 params=140651
 99 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
100 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
101 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
102 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.8 val_loss=0.781037 params=1077579
103 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.7875 val_loss=0.828938 params=1336459
104 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
105 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
106 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
107 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
108 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
109 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
110 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
111 confirmation swin3d_t complete | 64px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
112 confirmation swin3d_t complete | 32px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
113 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
114 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
115 confirmation custom_64_64_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.125 val_loss=2.3843 params=1336011
116 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Pending dependent stages (not yet resolved): temporal
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (4,582,062,066 bytes; 2798 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Experiment 113/142 campaign maximum | temporal: custom_64_96_64 64px, 16f/4fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_083108_994228_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Caching 908 train/validation clips (test excluded)...
  Cached 50/908
  Cached 100/908
  Cached 150/908
  Cached 200/908
  Cached 250/908
  Cached 300/908
  Cached 350/908
  Cached 400/908
  Cached 450/908
  Cached 500/908
  Cached 550/908
  Cached 600/908
  Cached 650/908
  Cached 700/908
  Cached 750/908
  Cached 800/908
  Cached 850/908
  Cached 900/908
  Cached 908/908
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 113/142 maximum | temporal

Experiment 113/142 maximum | temporal: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_083108_994228_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_083108_994228_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_083108_994228_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.7937 val_loss=0.6774 selected_epoch=254 time=2243s
{'class_name': 'Begging', 'precision': 0.76, 'recall': 0.95, 'f1': 0.8444444444444444, 'support': 20, 'tp': 19, 'fp': 6, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.5833333333333334, 'recall': 0.5384615384615384, 'f1': 0.56, 'support': 13, 'tp': 7, 'fp': 5, 'fn': 6}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.5, 'f1': 0.6666666666666666, 'support': 14, 'tp': 7, 'fp': 0, 'fn': 7}
{'class_name': 'Harassment', 'precision': 0.875, 'recall': 0.5, 'f1': 0.6363636363636364, 'support': 14, 'tp': 7, 'fp': 1, 'fn': 7}
{'class_name': 'Hijack', 'precision': 0.75, 'recall': 0.75, 'f1': 0.75, 'support': 12, 'tp': 9, 'fp': 3, 'fn': 3}
{'class_name': 'Knife Hazard', 'precision': 0.6470588235294118, 'recall': 0.7333333333333333, 'f1': 0.6875, 'support': 15, 'tp': 11, 'fp': 6, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.8947368421052632, 'recall': 1.0, 'f1': 0.9444444444444444, 'support': 17, 'tp': 17, 'fp': 2, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.7, 'recall': 0.8235294117647058, 'f1': 0.7567567567567568, 'support': 17, 'tp': 14, 'fp': 6, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.75, 'recall': 0.8181818181818182, 'f1': 0.782608695652174, 'support': 11, 'tp': 9, 'fp': 3, 'fn': 2}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_083108_994228_abnormal-activities-dataset_capacity/summary.json
Experiment 114/142 campaign maximum | temporal: custom_64_96_64 64px, 16f/8fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_090908_427142_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Caching 908 train/validation clips (test excluded)...
  Cached 50/908
  Cached 100/908
  Cached 150/908
  Cached 200/908
  Cached 250/908
  Cached 300/908
  Cached 350/908
  Cached 400/908
  Cached 450/908
  Cached 500/908
  Cached 550/908
  Cached 600/908
  Cached 650/908
  Cached 700/908
  Cached 750/908
  Cached 800/908
  Cached 850/908
  Cached 900/908
  Cached 908/908
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 114/142 maximum | temporal

Experiment 114/142 maximum | temporal: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_090908_427142_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_090908_427142_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_090908_427142_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9312 val_loss=0.3874 selected_epoch=253 time=2259s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.7857142857142857, 'recall': 0.8461538461538461, 'f1': 0.8148148148148148, 'support': 13, 'tp': 11, 'fp': 3, 'fn': 2}
{'class_name': 'Fight', 'precision': 0.8666666666666667, 'recall': 0.9285714285714286, 'f1': 0.896551724137931, 'support': 14, 'tp': 13, 'fp': 2, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.7692307692307693, 'recall': 0.6666666666666666, 'f1': 0.7142857142857143, 'support': 15, 'tp': 10, 'fp': 3, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_090908_427142_abnormal-activities-dataset_capacity/summary.json
Experiment 115/142 campaign maximum | temporal: custom_64_96_64 64px, 16f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_094717_883493_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Caching 908 train/validation clips (test excluded)...
  Cached 50/908
  Cached 100/908
  Cached 150/908
  Cached 200/908
  Cached 250/908
  Cached 300/908
  Cached 350/908
  Cached 400/908
  Cached 450/908
  Cached 500/908
  Cached 550/908
  Cached 600/908
  Cached 650/908
  Cached 700/908
  Cached 750/908
  Cached 800/908
  Cached 850/908
  Cached 900/908
  Cached 908/908
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 115/142 maximum | temporal

Experiment 115/142 maximum | temporal: custom_64_96_64 | params=1,077,579
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_094717_883493_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_094717_883493_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_094717_883493_abnormal-activities-dataset_capacity/models/custom_64_96_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9438 val_loss=0.3436 selected_epoch=256 time=2265s
{'class_name': 'Begging', 'precision': 0.95, 'recall': 0.95, 'f1': 0.95, 'support': 20, 'tp': 19, 'fp': 1, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8333333333333334, 'recall': 0.7692307692307693, 'f1': 0.8, 'support': 13, 'tp': 10, 'fp': 2, 'fn': 3}
{'class_name': 'Fight', 'precision': 0.9333333333333333, 'recall': 1.0, 'f1': 0.9655172413793104, 'support': 14, 'tp': 14, 'fp': 1, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9166666666666666, 'recall': 0.9166666666666666, 'f1': 0.9166666666666666, 'support': 12, 'tp': 11, 'fp': 1, 'fn': 1}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.9411764705882353, 'f1': 0.9696969696969697, 'support': 17, 'tp': 16, 'fp': 0, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.9166666666666666, 'recall': 1.0, 'f1': 0.9565217391304348, 'support': 11, 'tp': 11, 'fp': 1, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_094717_883493_abnormal-activities-dataset_capacity/summary.json
Experiment 116/142 campaign maximum | temporal: custom_96_64_64 64px, 16f/4fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_102531_219872_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 116/142 maximum | temporal

Experiment 116/142 maximum | temporal: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_102531_219872_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_102531_219872_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_102531_219872_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.5813 val_loss=1.2651 selected_epoch=87 time=976s
{'class_name': 'Begging', 'precision': 0.5, 'recall': 0.8, 'f1': 0.6153846153846154, 'support': 20, 'tp': 16, 'fp': 16, 'fn': 4}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0.5, 'recall': 0.07142857142857142, 'f1': 0.125, 'support': 14, 'tp': 1, 'fp': 1, 'fn': 13}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0.75, 'recall': 0.5, 'f1': 0.6, 'support': 12, 'tp': 6, 'fp': 2, 'fn': 6}
{'class_name': 'Knife Hazard', 'precision': 0.37037037037037035, 'recall': 0.6666666666666666, 'f1': 0.47619047619047616, 'support': 15, 'tp': 10, 'fp': 17, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.7894736842105263, 'recall': 0.8823529411764706, 'f1': 0.8333333333333334, 'support': 17, 'tp': 15, 'fp': 4, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.5555555555555556, 'recall': 0.8823529411764706, 'f1': 0.6818181818181818, 'support': 17, 'tp': 15, 'fp': 12, 'fn': 2}
{'class_name': 'Property Damage', 'precision': 0.6666666666666666, 'recall': 1.0, 'f1': 0.8, 'support': 14, 'tp': 14, 'fp': 7, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.6, 'recall': 0.6923076923076923, 'f1': 0.6428571428571429, 'support': 13, 'tp': 9, 'fp': 6, 'fn': 4}
{'class_name': 'Terrorism', 'precision': 0.7777777777777778, 'recall': 0.6363636363636364, 'f1': 0.7, 'support': 11, 'tp': 7, 'fp': 2, 'fn': 4}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_102531_219872_abnormal-activities-dataset_capacity/summary.json
Experiment 117/142 campaign maximum | temporal: custom_96_64_64 64px, 16f/8fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_104157_515351_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 117/142 maximum | temporal

Experiment 117/142 maximum | temporal: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_104157_515351_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_104157_515351_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_104157_515351_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.3981 selected_epoch=256 time=2250s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8, 'recall': 0.9230769230769231, 'f1': 0.8571428571428571, 'support': 13, 'tp': 12, 'fp': 3, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 12, 'tp': 12, 'fp': 0, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.9230769230769231, 'recall': 0.8, 'f1': 0.8571428571428571, 'support': 15, 'tp': 12, 'fp': 1, 'fn': 3}
{'class_name': 'Normal Videos', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Pollution', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_104157_515351_abnormal-activities-dataset_capacity/summary.json
Experiment 118/142 campaign maximum | temporal: custom_96_64_64 64px, 16f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_111937_444155_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 118/142 maximum | temporal

Experiment 118/142 maximum | temporal: custom_96_64_64 | params=1,007,307
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_111937_444155_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_111937_444155_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_111937_444155_abnormal-activities-dataset_capacity/models/custom_96_64_64/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.4812 val_loss=1.3246 selected_epoch=113 time=1205s
{'class_name': 'Begging', 'precision': 0.4666666666666667, 'recall': 0.7, 'f1': 0.56, 'support': 20, 'tp': 14, 'fp': 16, 'fn': 6}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0.8333333333333334, 'recall': 0.35714285714285715, 'f1': 0.5, 'support': 14, 'tp': 5, 'fp': 1, 'fn': 9}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0.4, 'recall': 0.3333333333333333, 'f1': 0.36363636363636365, 'support': 12, 'tp': 4, 'fp': 6, 'fn': 8}
{'class_name': 'Knife Hazard', 'precision': 0.4, 'recall': 0.6666666666666666, 'f1': 0.5, 'support': 15, 'tp': 10, 'fp': 15, 'fn': 5}
{'class_name': 'Normal Videos', 'precision': 0.6111111111111112, 'recall': 0.6470588235294118, 'f1': 0.6285714285714286, 'support': 17, 'tp': 11, 'fp': 7, 'fn': 6}
{'class_name': 'Pollution', 'precision': 0.34375, 'recall': 0.6470588235294118, 'f1': 0.4489795918367347, 'support': 17, 'tp': 11, 'fp': 21, 'fn': 6}
{'class_name': 'Property Damage', 'precision': 0.5833333333333334, 'recall': 1.0, 'f1': 0.7368421052631579, 'support': 14, 'tp': 14, 'fp': 10, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.5, 'recall': 0.38461538461538464, 'f1': 0.43478260869565216, 'support': 13, 'tp': 5, 'fp': 5, 'fn': 8}
{'class_name': 'Terrorism', 'precision': 0.6, 'recall': 0.2727272727272727, 'f1': 0.375, 'support': 11, 'tp': 3, 'fp': 2, 'fn': 8}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_111937_444155_abnormal-activities-dataset_capacity/summary.json
Experiment 119/142 campaign maximum | temporal: custom_64_64_128 64px, 16f/4fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_113952_684672_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 119/142 maximum | temporal

Experiment 119/142 maximum | temporal: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_113952_684672_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_113952_684672_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_113952_684672_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.8750 val_loss=0.5954 selected_epoch=256 time=2372s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.85, 'f1': 0.918918918918919, 'support': 20, 'tp': 17, 'fp': 0, 'fn': 3}
{'class_name': 'Drunkenness', 'precision': 0.6, 'recall': 0.9230769230769231, 'f1': 0.7272727272727273, 'support': 13, 'tp': 12, 'fp': 8, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.8461538461538461, 'recall': 0.7857142857142857, 'f1': 0.8148148148148148, 'support': 14, 'tp': 11, 'fp': 2, 'fn': 3}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.8333333333333334, 'recall': 0.8333333333333334, 'f1': 0.8333333333333334, 'support': 12, 'tp': 10, 'fp': 2, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.8181818181818182, 'recall': 0.6, 'f1': 0.6923076923076923, 'support': 15, 'tp': 9, 'fp': 2, 'fn': 6}
{'class_name': 'Normal Videos', 'precision': 0.9375, 'recall': 0.8823529411764706, 'f1': 0.9090909090909091, 'support': 17, 'tp': 15, 'fp': 1, 'fn': 2}
{'class_name': 'Pollution', 'precision': 0.8421052631578947, 'recall': 0.9411764705882353, 'f1': 0.8888888888888888, 'support': 17, 'tp': 16, 'fp': 3, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.8333333333333334, 'recall': 0.9090909090909091, 'f1': 0.8695652173913043, 'support': 11, 'tp': 10, 'fp': 2, 'fn': 1}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_113952_684672_abnormal-activities-dataset_capacity/summary.json
Experiment 120/142 campaign maximum | temporal: custom_64_64_128 64px, 16f/8fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_121934_946063_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 120/142 maximum | temporal

Experiment 120/142 maximum | temporal: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_121934_946063_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_121934_946063_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_121934_946063_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.2688 val_loss=1.9774 selected_epoch=14 time=590s
{'class_name': 'Begging', 'precision': 0.2289156626506024, 'recall': 0.95, 'f1': 0.36893203883495146, 'support': 20, 'tp': 19, 'fp': 64, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0.14285714285714285, 'recall': 0.16666666666666666, 'f1': 0.15384615384615385, 'support': 12, 'tp': 2, 'fp': 12, 'fn': 10}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0.45161290322580644, 'recall': 0.8235294117647058, 'f1': 0.5833333333333334, 'support': 17, 'tp': 14, 'fp': 17, 'fn': 3}
{'class_name': 'Pollution', 'precision': 0.25, 'recall': 0.47058823529411764, 'f1': 0.32653061224489793, 'support': 17, 'tp': 8, 'fp': 24, 'fn': 9}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_121934_946063_abnormal-activities-dataset_capacity/summary.json
Experiment 121/142 campaign maximum | temporal: custom_64_64_128 64px, 16f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_122935_351542_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 121/142 maximum | temporal

Experiment 121/142 maximum | temporal: custom_64_64_128 | params=1,336,459
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_122935_351542_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_122935_351542_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_122935_351542_abnormal-activities-dataset_capacity/models/custom_64_64_128/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.3620 selected_epoch=250 time=2359s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 0.8571428571428571, 'recall': 0.9230769230769231, 'f1': 0.8888888888888888, 'support': 13, 'tp': 12, 'fp': 2, 'fn': 1}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Hijack', 'precision': 0.9090909090909091, 'recall': 0.8333333333333334, 'f1': 0.8695652173913043, 'support': 12, 'tp': 10, 'fp': 1, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.9285714285714286, 'recall': 0.8666666666666667, 'f1': 0.896551724137931, 'support': 15, 'tp': 13, 'fp': 1, 'fn': 2}
{'class_name': 'Normal Videos', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 13, 'tp': 13, 'fp': 0, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_122935_351542_abnormal-activities-dataset_capacity/summary.json
Experiment 122/142 campaign maximum | temporal: r3d_18 64px, 16f/4fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_130904_470949_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 122/142 maximum | temporal

Experiment 122/142 maximum | temporal: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_130904_470949_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_130904_470949_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_130904_470949_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9187 val_loss=0.2609 selected_epoch=29 time=128s
{'class_name': 'Begging', 'precision': 0.9473684210526315, 'recall': 0.9, 'f1': 0.9230769230769231, 'support': 20, 'tp': 18, 'fp': 1, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.8666666666666667, 'recall': 0.9285714285714286, 'f1': 0.896551724137931, 'support': 14, 'tp': 13, 'fp': 2, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.8125, 'recall': 0.9285714285714286, 'f1': 0.8666666666666667, 'support': 14, 'tp': 13, 'fp': 3, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9090909090909091, 'recall': 0.8333333333333334, 'f1': 0.8695652173913043, 'support': 12, 'tp': 10, 'fp': 1, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.9333333333333333, 'recall': 0.9333333333333333, 'f1': 0.9333333333333333, 'support': 15, 'tp': 14, 'fp': 1, 'fn': 1}
{'class_name': 'Normal Videos', 'precision': 0.85, 'recall': 1.0, 'f1': 0.918918918918919, 'support': 17, 'tp': 17, 'fp': 3, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 0.8235294117647058, 'f1': 0.9032258064516129, 'support': 17, 'tp': 14, 'fp': 0, 'fn': 3}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Terrorism', 'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 11, 'tp': 11, 'fp': 2, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_130904_470949_abnormal-activities-dataset_capacity/summary.json
Experiment 123/142 campaign maximum | temporal: r3d_18 64px, 16f/8fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131126_005363_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 123/142 maximum | temporal

Experiment 123/142 maximum | temporal: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131126_005363_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131126_005363_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131126_005363_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9062 val_loss=0.3261 selected_epoch=43 time=136s
{'class_name': 'Begging', 'precision': 0.9473684210526315, 'recall': 0.9, 'f1': 0.9230769230769231, 'support': 20, 'tp': 18, 'fp': 1, 'fn': 2}
{'class_name': 'Drunkenness', 'precision': 0.75, 'recall': 0.9230769230769231, 'f1': 0.8275862068965517, 'support': 13, 'tp': 12, 'fp': 4, 'fn': 1}
{'class_name': 'Fight', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Harassment', 'precision': 1.0, 'recall': 0.7142857142857143, 'f1': 0.8333333333333334, 'support': 14, 'tp': 10, 'fp': 0, 'fn': 4}
{'class_name': 'Hijack', 'precision': 1.0, 'recall': 0.8333333333333334, 'f1': 0.9090909090909091, 'support': 12, 'tp': 10, 'fp': 0, 'fn': 2}
{'class_name': 'Knife Hazard', 'precision': 0.9166666666666666, 'recall': 0.7333333333333333, 'f1': 0.8148148148148148, 'support': 15, 'tp': 11, 'fp': 1, 'fn': 4}
{'class_name': 'Normal Videos', 'precision': 0.85, 'recall': 1.0, 'f1': 0.918918918918919, 'support': 17, 'tp': 17, 'fp': 3, 'fn': 0}
{'class_name': 'Pollution', 'precision': 0.9411764705882353, 'recall': 0.9411764705882353, 'f1': 0.9411764705882353, 'support': 17, 'tp': 16, 'fp': 1, 'fn': 1}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 0.9285714285714286, 'recall': 1.0, 'f1': 0.9629629629629629, 'support': 13, 'tp': 13, 'fp': 1, 'fn': 0}
{'class_name': 'Terrorism', 'precision': 0.7857142857142857, 'recall': 1.0, 'f1': 0.88, 'support': 11, 'tp': 11, 'fp': 3, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131126_005363_abnormal-activities-dataset_capacity/summary.json
Experiment 124/142 campaign maximum | temporal: r3d_18 64px, 16f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131354_853986_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 124/142 maximum | temporal

Experiment 124/142 maximum | temporal: r3d_18 | params=33,171,915
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131354_853986_abnormal-activities-dataset_capacity/models/r3d_18/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131354_853986_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131354_853986_abnormal-activities-dataset_capacity/models/r3d_18/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.9625 val_loss=0.2684 selected_epoch=103 time=246s
{'class_name': 'Begging', 'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 20, 'tp': 19, 'fp': 0, 'fn': 1}
{'class_name': 'Drunkenness', 'precision': 1.0, 'recall': 0.8461538461538461, 'f1': 0.9166666666666666, 'support': 13, 'tp': 11, 'fp': 0, 'fn': 2}
{'class_name': 'Fight', 'precision': 1.0, 'recall': 0.9285714285714286, 'f1': 0.9629629629629629, 'support': 14, 'tp': 13, 'fp': 0, 'fn': 1}
{'class_name': 'Harassment', 'precision': 0.9285714285714286, 'recall': 0.9285714285714286, 'f1': 0.9285714285714286, 'support': 14, 'tp': 13, 'fp': 1, 'fn': 1}
{'class_name': 'Hijack', 'precision': 0.9230769230769231, 'recall': 1.0, 'f1': 0.96, 'support': 12, 'tp': 12, 'fp': 1, 'fn': 0}
{'class_name': 'Knife Hazard', 'precision': 0.8333333333333334, 'recall': 1.0, 'f1': 0.9090909090909091, 'support': 15, 'tp': 15, 'fp': 3, 'fn': 0}
{'class_name': 'Normal Videos', 'precision': 0.9444444444444444, 'recall': 1.0, 'f1': 0.9714285714285714, 'support': 17, 'tp': 17, 'fp': 1, 'fn': 0}
{'class_name': 'Pollution', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 17, 'tp': 17, 'fp': 0, 'fn': 0}
{'class_name': 'Property Damage', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 14, 'tp': 14, 'fp': 0, 'fn': 0}
{'class_name': 'Robbery', 'precision': 1.0, 'recall': 0.9230769230769231, 'f1': 0.96, 'support': 13, 'tp': 12, 'fp': 0, 'fn': 1}
{'class_name': 'Terrorism', 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 11, 'tp': 11, 'fp': 0, 'fn': 0}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131354_853986_abnormal-activities-dataset_capacity/summary.json
Experiment 125/142 campaign maximum | temporal: swin3d_t 64px, 16f/4fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131814_506698_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 125/142 maximum | temporal

Experiment 125/142 maximum | temporal: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131814_506698_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131814_506698_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131814_506698_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=60 time=345s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_131814_506698_abnormal-activities-dataset_capacity/summary.json
Experiment 126/142 campaign maximum | temporal: swin3d_t 64px, 16f/8fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132413_377286_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 126/142 maximum | temporal

Experiment 126/142 maximum | temporal: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132413_377286_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132413_377286_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132413_377286_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3843 selected_epoch=43 time=277s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132413_377286_abnormal-activities-dataset_capacity/summary.json
Experiment 127/142 campaign maximum | temporal: swin3d_t 64px, 16f/16fps, seed 42
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132904_625908_abnormal-activities-dataset_capacity
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 127/142 maximum | temporal

Experiment 127/142 maximum | temporal: swin3d_t | params=27,858,929
Saved training curves to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132904_625908_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/training_curves.png
📊 Saved confusion matrix to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132904_625908_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.png and /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132904_625908_abnormal-activities-dataset_capacity/models/swin3d_t/metrics/validation_confusion_matrix.json
Prediction examples (validation): limit=0 per category; correct=0, incorrect=0
val_acc=0.1250 val_loss=2.3842 selected_epoch=60 time=346s
{'class_name': 'Begging', 'precision': 0.125, 'recall': 1.0, 'f1': 0.2222222222222222, 'support': 20, 'tp': 20, 'fp': 140, 'fn': 0}
{'class_name': 'Drunkenness', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Fight', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Harassment', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Hijack', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 12, 'tp': 0, 'fp': 0, 'fn': 12}
{'class_name': 'Knife Hazard', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 15, 'tp': 0, 'fp': 0, 'fn': 15}
{'class_name': 'Normal Videos', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Pollution', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 17, 'tp': 0, 'fp': 0, 'fn': 17}
{'class_name': 'Property Damage', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 14, 'tp': 0, 'fp': 0, 'fn': 14}
{'class_name': 'Robbery', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 13, 'tp': 0, 'fp': 0, 'fn': 13}
{'class_name': 'Terrorism', 'precision': 0, 'recall': 0.0, 'f1': 0.0, 'support': 11, 'tp': 0, 'fp': 0, 'fn': 11}

Full results saved to /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_132904_625908_abnormal-activities-dataset_capacity/summary.json

Full results: 131 completed; 0 failed/pending
  1 refinement custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.99375 val_loss=0.29771 params=1007307
  2 confirmation custom_128_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.0979638 params=1342667
  3 confirmation custom_96_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.144812 params=1007307
  4 refinement custom_32_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.170426 params=557771
  5 confirmation custom_64_64_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.174946 params=1336011
  6 refinement custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.179411 params=1077579
  7 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  8 confirmation custom_64_96_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.193503 params=1077579
  9 refinement custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.241714 params=1077579
 10 refinement custom_64_128_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.258579 params=1483211
 11 refinement custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.290738 params=1336459
 12 refinement custom_128_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.149044 params=1342667
 13 refinement custom_64_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.155748 params=560875
 14 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
 15 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
 16 refinement custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.184906 params=1336459
 17 refinement custom_64_64_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.197078 params=1004203
 18 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.98125 val_loss=0.204597 params=1336459
 19 confirmation custom_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.205548 params=745675
 20 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
 21 confirmation custom_96_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.229499 params=1007307
 22 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
 23 confirmation custom_64_96_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.289887 params=1077579
 24 confirmation custom_64_64_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.291864 params=1336459
 25 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
 26 confirmation custom_96_96_96 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.313502 params=1671467
 27 refinement custom_128_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.317599 params=1342667
 28 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 29 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 30 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 31 confirmation custom_64_64_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.360393 params=1336459
 32 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 33 refinement custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.104137 params=1007307
 34 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 35 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 36 refinement custom_64_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.175208 params=487499
 37 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.180159 params=1007307
 38 confirmation custom_128_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.183511 params=1342667
 39 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 40 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.189062 params=1077579
 41 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 42 refinement custom_64_128_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.215485 params=1483211
 43 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 44 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 45 confirmation custom_48_48_48 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.219522 params=421019
 46 refinement custom_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.220924 params=265707
 47 refinement custom_64_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.230771 params=560875
 48 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 49 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.241667 params=1007307
 50 confirmation custom_96_96_96 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.258062 params=1671467
 51 confirmation custom_128_128_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.275204 params=2965899
 52 refinement custom_64_64_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.277994 params=1004203
 53 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.291667 params=1077579
 54 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 55 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 56 refinement custom_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.337258 params=262603
 57 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 58 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 59 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 60 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 61 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 62 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.96875 val_loss=0.227579 params=1336459
 63 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 64 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.96875 val_loss=0.270876 params=1336459
 65 confirmation custom_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.316626 params=745675
 66 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 67 refinement custom_64_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.385308 params=487499
 68 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 69 refinement custom_32_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.407691 params=557771
 70 confirmation custom_128_128_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.181375 params=2965899
 71 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 72 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 73 temporal r3d_18 complete | 64px 16f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.268378 params=33171915
 74 confirmation custom_32_32_32 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.291422 params=188523
 75 temporal custom_64_64_128 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.361994 params=1336459
 76 confirmation custom_48_48_48 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.377144 params=421019
 77 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 78 temporal custom_96_64_64 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.398109 params=1007307
 79 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 80 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 81 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.95625 val_loss=0.229204 params=1077579
 82 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 83 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 84 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 85 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 86 refinement custom_16_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.3268 params=140651
 87 temporal custom_64_96_64 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.343641 params=1077579
 88 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 89 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.9375 val_loss=0.279749 params=1007307
 90 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 91 temporal custom_64_96_64 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.387362 params=1077579
 92 refinement custom_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.925 val_loss=0.520644 params=262603
 93 temporal r3d_18 complete | 64px 16f/4fps seed=42 wd=0 lr=0.003 val_acc=0.91875 val_loss=0.260925 params=33171915
 94 refinement custom_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9125 val_loss=0.568474 params=265707
 95 temporal r3d_18 complete | 64px 16f/8fps seed=42 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.3261 params=33171915
 96 confirmation r3d_18 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.368961 params=33171915
 97 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 98 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 99 confirmation r3d_18 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.467042 params=33171915
100 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
101 temporal custom_64_64_128 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.875 val_loss=0.595445 params=1336459
102 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.85 val_loss=0.592975 params=1007307
103 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
104 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
105 confirmation custom_32_32_32 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.640571 params=188523
106 refinement custom_16_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8375 val_loss=0.656602 params=140651
107 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
108 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
109 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
110 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.8 val_loss=0.781037 params=1077579
111 temporal custom_64_96_64 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.79375 val_loss=0.677362 params=1077579
112 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.7875 val_loss=0.828938 params=1336459
113 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
114 temporal custom_96_64_64 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.58125 val_loss=1.26508 params=1007307
115 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
116 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
117 temporal custom_96_64_64 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.48125 val_loss=1.32461 params=1007307
118 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
119 temporal custom_64_64_128 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.26875 val_loss=1.97741 params=1336459
120 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
121 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
122 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
123 confirmation swin3d_t complete | 64px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
124 temporal swin3d_t complete | 64px 16f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
125 confirmation swin3d_t complete | 32px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
126 temporal swin3d_t complete | 64px 16f/4fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38421 params=27858929
127 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
128 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
129 temporal swin3d_t complete | 64px 16f/8fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38426 params=27858929
130 confirmation custom_64_64_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.125 val_loss=2.3843 params=1336011
131 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (6,720,085,804 bytes; 3161 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.

Full results: 131 completed; 0 failed/pending
  1 refinement custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.99375 val_loss=0.29771 params=1007307
  2 confirmation custom_128_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.0979638 params=1342667
  3 confirmation custom_96_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.144812 params=1007307
  4 refinement custom_32_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.170426 params=557771
  5 confirmation custom_64_64_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.174946 params=1336011
  6 refinement custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.179411 params=1077579
  7 flat custom_48_48_48 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.189388 params=421019
  8 confirmation custom_64_96_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.193503 params=1077579
  9 refinement custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.241714 params=1077579
 10 refinement custom_64_128_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.258579 params=1483211
 11 refinement custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9875 val_loss=0.290738 params=1336459
 12 refinement custom_128_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.149044 params=1342667
 13 refinement custom_64_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.155748 params=560875
 14 flat custom_64_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.167842 params=1336011
 15 flat custom_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.17606 params=1671467
 16 refinement custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.184906 params=1336459
 17 refinement custom_64_64_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.197078 params=1004203
 18 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.98125 val_loss=0.204597 params=1336459
 19 confirmation custom_64_64_64 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.205548 params=745675
 20 flat custom_128_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.216447 params=4146059
 21 confirmation custom_96_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.229499 params=1007307
 22 flat custom_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.272935 params=1007531
 23 confirmation custom_64_96_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.289887 params=1077579
 24 confirmation custom_64_64_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.291864 params=1336459
 25 flat custom_48_48_48 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.297459 params=421019
 26 confirmation custom_96_96_96 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.313502 params=1671467
 27 refinement custom_128_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.317599 params=1342667
 28 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 29 flat custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.33101 params=745675
 30 flat custom_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.347885 params=605579
 31 confirmation custom_64_64_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.360393 params=1336459
 32 flat custom_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.98125 val_loss=0.362039 params=1007531
 33 refinement custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.104137 params=1007307
 34 flat custom_64_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.145755 params=1040843
 35 flat custom_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.16665 params=1785739
 36 refinement custom_64_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.175208 params=487499
 37 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.180159 params=1007307
 38 confirmation custom_128_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.183511 params=1342667
 39 flat custom_96_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.184878 params=2335403
 40 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.189062 params=1077579
 41 flat custom_32_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.213467 params=262379
 42 refinement custom_64_128_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.215485 params=1483211
 43 flat custom_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.216109 params=450507
 44 flat custom_64_64_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.217643 params=745675
 45 confirmation custom_48_48_48 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.219522 params=421019
 46 refinement custom_64_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.220924 params=265707
 47 refinement custom_64_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.230771 params=560875
 48 flat custom_80_80_80 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.237882 params=1162491
 49 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.241667 params=1007307
 50 confirmation custom_96_96_96 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.258062 params=1671467
 51 confirmation custom_128_128_128 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.975 val_loss=0.275204 params=2965899
 52 refinement custom_64_64_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.277994 params=1004203
 53 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.975 val_loss=0.291667 params=1077579
 54 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 55 flat custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.306628 params=2965899
 56 refinement custom_32_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.337258 params=262603
 57 flat custom_32_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.344111 params=262379
 58 flat custom_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.356938 params=1785739
 59 flat custom_64 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.975 val_loss=0.370779 params=155339
 60 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.975 val_loss=0.431247 params=745675
 61 flat custom_128_128_128 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.199426 params=2965899
 62 weight_decay custom_64_64_128 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.96875 val_loss=0.227579 params=1336459
 63 flat custom_32_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.256615 params=188523
 64 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.0001 lr=0.001 val_acc=0.96875 val_loss=0.270876 params=1336459
 65 confirmation custom_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.316626 params=745675
 66 flat custom_80_80_80 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.324572 params=1162491
 67 refinement custom_64_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.385308 params=487499
 68 flat custom_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.38722 params=605579
 69 refinement custom_32_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.96875 val_loss=0.407691 params=557771
 70 confirmation custom_128_128_128 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.181375 params=2965899
 71 flat custom_96_96_96_96 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.20954 params=2335403
 72 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.258612 params=2965899
 73 temporal r3d_18 complete | 64px 16f/16fps seed=42 wd=0 lr=0.003 val_acc=0.9625 val_loss=0.268378 params=33171915
 74 confirmation custom_32_32_32 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.291422 params=188523
 75 temporal custom_64_64_128 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.361994 params=1336459
 76 confirmation custom_48_48_48 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.377144 params=421019
 77 flat custom_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.393042 params=450507
 78 temporal custom_96_64_64 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.9625 val_loss=0.398109 params=1007307
 79 calibration custom_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9625 val_loss=0.548138 params=2965899
 80 flat custom_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.228611 params=1040843
 81 weight_decay custom_64_96_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.95625 val_loss=0.229204 params=1077579
 82 flat custom_96_96_96 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95625 val_loss=0.416278 params=1671467
 83 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 84 references r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.95 val_loss=0.259127 params=33171915
 85 flat custom_128_128_128_128 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.95 val_loss=0.407025 params=4146059
 86 refinement custom_16_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.3268 params=140651
 87 temporal custom_64_96_64 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.94375 val_loss=0.343641 params=1077579
 88 calibration custom_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.94375 val_loss=0.434665 params=745675
 89 weight_decay custom_96_64_64 complete | 32px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.9375 val_loss=0.279749 params=1007307
 90 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.276306 params=33171915
 91 temporal custom_64_96_64 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.93125 val_loss=0.387362 params=1077579
 92 refinement custom_32_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.925 val_loss=0.520644 params=262603
 93 temporal r3d_18 complete | 64px 16f/4fps seed=42 wd=0 lr=0.003 val_acc=0.91875 val_loss=0.260925 params=33171915
 94 refinement custom_64_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.9125 val_loss=0.568474 params=265707
 95 temporal r3d_18 complete | 64px 16f/8fps seed=42 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.3261 params=33171915
 96 confirmation r3d_18 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.90625 val_loss=0.368961 params=33171915
 97 calibration r3d_18 complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.9 val_loss=0.331567 params=33171915
 98 references r3d_18 complete | 32px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.36439 params=33171915
 99 confirmation r3d_18 complete | 32px 8f/16fps seed=2026 wd=0 lr=0.003 val_acc=0.8875 val_loss=0.467042 params=33171915
100 flat custom_32_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8875 val_loss=0.507413 params=114667
101 temporal custom_64_64_128 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.875 val_loss=0.595445 params=1336459
102 weight_decay custom_96_64_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.85 val_loss=0.592975 params=1007307
103 flat custom_32 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.85 val_loss=0.745841 params=40811
104 flat custom_32_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.619465 params=188523
105 confirmation custom_32_32_32 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.84375 val_loss=0.640571 params=188523
106 refinement custom_16_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.8375 val_loss=0.656602 params=140651
107 flat custom_16_16_16 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.825 val_loss=0.687246 params=48187
108 flat custom_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.81875 val_loss=0.740383 params=155339
109 flat custom_32_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.80625 val_loss=0.810132 params=114667
110 weight_decay custom_64_96_64 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.8 val_loss=0.781037 params=1077579
111 temporal custom_64_96_64 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.79375 val_loss=0.677362 params=1077579
112 weight_decay custom_64_64_128 complete | 64px 8f/16fps seed=42 wd=0.001 lr=0.001 val_acc=0.7875 val_loss=0.828938 params=1336459
113 flat custom_16_16_16 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.70625 val_loss=0.916551 params=48187
114 temporal custom_96_64_64 complete | 64px 16f/4fps seed=42 wd=0 lr=0.001 val_acc=0.58125 val_loss=1.26508 params=1007307
115 flat custom_32 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.56875 val_loss=1.13164 params=40811
116 flat custom_8_8_8 complete | 32px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.5375 val_loss=1.19879 params=12579
117 temporal custom_96_64_64 complete | 64px 16f/16fps seed=42 wd=0 lr=0.001 val_acc=0.48125 val_loss=1.32461 params=1007307
118 flat custom_8_8_8 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.4625 val_loss=1.35899 params=12579
119 temporal custom_64_64_128 complete | 64px 16f/8fps seed=42 wd=0 lr=0.001 val_acc=0.26875 val_loss=1.97741 params=1336459
120 references swin3d_t complete | 32px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
121 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
122 references swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38418 params=27858929
123 confirmation swin3d_t complete | 64px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
124 temporal swin3d_t complete | 64px 16f/16fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
125 confirmation swin3d_t complete | 32px 8f/16fps seed=2026 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38419 params=27858929
126 temporal swin3d_t complete | 64px 16f/4fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38421 params=27858929
127 flat custom_64_64_64_64_64 complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38421 params=1336011
128 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.003 val_acc=0.125 val_loss=2.38424 params=27858929
129 temporal swin3d_t complete | 64px 16f/8fps seed=42 wd=0 lr=0.01 val_acc=0.125 val_loss=2.38426 params=27858929
130 confirmation custom_64_64_64_64_64 complete | 64px 8f/16fps seed=2026 wd=0 lr=0.001 val_acc=0.125 val_loss=2.3843 params=1336011
131 calibration swin3d_t complete | 64px 8f/16fps seed=42 wd=0 lr=0.001 val_acc=0.125 val_loss=2.38434 params=27858929
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/capacity_archives/20261010_144535_699826_abnormal-activities-dataset_capacity/artifacts.zip (6,720,278,948 bytes; 3166 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
Capacity search complete. Validation-only top three: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/studies/20261010_144535_699826_abnormal-activities-dataset_capacity/selected_config.json
Verified ZIP: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/notebook_studies/capacity_20261010_144521_885468/artifacts.zip (6,720,283,278 bytes; 3170 files)
Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.
2/2 Validate transferred LR/WD (up to 3 jobs), then eight fresh comparison trainings; validation freeze, test, examples and ZIP
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
$ /usr/bin/python3 -m src.experiments --config /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/notebook_studies/comparison_20261011_140033_909842/resolved_model_comparison.json --list-plan
AAD model-family comparison: 8 declared runs; no Cartesian expansion.
Dataset=/root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset; split=/content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/studies/20261010_144535_699826_abnormal-activities-dataset_capacity/split_manifest.json; split_seed=42
Fixed: epochs<=512, patience=32, batch=32, lr=0.001, scheduler=reduce_on_plateau
Cache processed clips in RAM: True
Before comparison: at most 3 validation-only native-input recipe jobs (128-cap/64-minimum), baseline then LR then WD. Printed LR/WD are the incumbent, not the final recipe. All eight models share the frozen result; search maximum 177 + recipe maximum 3 + comparison 8 = 188 trainings.
Minimum completed epochs before stopping: 128
Selection: validation accuracy, loss, parameters; test remains locked.
Single seed: no seed uncertainty or global-optimum/generalisation claim.
Eight fresh trainings; eight full test evaluations ONLY after all checkpoints and best custom are frozen.
Command: /usr/bin/python3 -m src.experiments --config /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/notebook_studies/comparison_20261011_140033_909842/resolved_model_comparison.json
01. screen: custom_top1 seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/01/config.json --model custom_top1 --candidates-config <study-run>/candidates.json
02. screen: custom_top2 seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/02/config.json --model custom_top2 --candidates-config <study-run>/candidates.json
03. screen: custom_top3 seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/03/config.json --model custom_top3 --candidates-config <study-run>/candidates.json
04. screen: paper_convlstm_adapted seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/04/config.json --model paper_convlstm_adapted
05. screen: r3d_18 seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/05/config.json --model r3d_18
06. screen: mc3_18 seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/06/config.json --model mc3_18
07. screen: swin3d_t seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/07/config.json --model swin3d_t
08. screen: swin3d_s seed=42 T=16 64x64 wd=0.0001 factor=architecture | leaf: python -m src.experiments --config <study-run>/jobs/08/config.json --model swin3d_s
Leaf JSON/commands are generated and recorded from this one file at execution.
Native PaperConvLSTM: T=50, 50x50, batch=1, separate/unranked; enabled=False.
$ /usr/bin/python3 -m src.experiments --config /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/notebook_studies/comparison_20261011_140033_909842/resolved_model_comparison.json
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']

Final-input FPS checks (validation only; matched selected checkpoint):
custom_64_96_64 | 4 FPS | train/val accuracy=0.9211/0.7937 | train/val loss=0.4286/0.6774 | prediction counts={'0': 25, '4': 12, '1': 12, '5': 17, '2': 7, '3': 8, '7': 20, '9': 14, '10': 12, '6': 19, '8': 14} | warnings=[]
custom_64_96_64 | 8 FPS | train/val accuracy=0.9933/0.9313 | train/val loss=0.1394/0.3874 | prediction counts={'0': 19, '4': 13, '1': 14, '5': 13, '2': 15, '3': 13, '9': 14, '6': 18, '7': 16, '8': 14, '10': 11} | warnings=[]
custom_64_96_64 | 16 FPS | train/val accuracy=1.0000/0.9437 | train/val loss=0.0116/0.3436 | prediction counts={'0': 20, '4': 12, '1': 12, '2': 15, '5': 14, '3': 13, '9': 14, '10': 12, '6': 18, '7': 16, '8': 14} | warnings=[]
custom_96_64_64 | 4 FPS | train/val accuracy=0.5628/0.5813 | train/val loss=1.1697/1.2651 | prediction counts={'0': 32, '5': 27, '6': 19, '2': 2, '9': 15, '7': 27, '8': 21, '4': 8, '10': 9} | warnings=[]
custom_96_64_64 | 8 FPS | train/val accuracy=1.0000/0.9625 | train/val loss=0.0738/0.3981 | prediction counts={'0': 19, '6': 17, '1': 15, '5': 13, '2': 14, '3': 14, '4': 12, '7': 18, '8': 14, '9': 13, '10': 11} | warnings=[]
custom_96_64_64 | 16 FPS | train/val accuracy=0.5495/0.4813 | train/val loss=1.1921/1.3246 | prediction counts={'0': 30, '2': 6, '5': 25, '4': 10, '7': 32, '9': 10, '6': 18, '8': 24, '10': 5} | warnings=[]
custom_64_64_128 | 4 FPS | train/val accuracy=0.9412/0.8750 | train/val loss=0.2720/0.5954 | prediction counts={'0': 17, '2': 13, '1': 20, '4': 12, '5': 11, '3': 13, '7': 19, '10': 12, '6': 16, '8': 14, '9': 13} | warnings=[]
custom_64_64_128 | 8 FPS | train/val accuracy=0.2754/0.2687 | train/val loss=1.8848/1.9774 | prediction counts={'0': 83, '6': 31, '7': 32, '4': 14} | warnings=[]
custom_64_64_128 | 16 FPS | train/val accuracy=0.9933/0.9625 | train/val loss=0.0752/0.3620 | prediction counts={'0': 19, '4': 11, '1': 14, '5': 14, '2': 14, '3': 14, '10': 13, '6': 17, '7': 17, '8': 14, '9': 13} | warnings=[]
r3d_18 | 4 FPS | train/val accuracy=0.9813/0.9187 | train/val loss=0.0583/0.2609 | prediction counts={'0': 19, '2': 15, '4': 11, '1': 12, '5': 15, '3': 16, '6': 20, '10': 13, '7': 14, '8': 14, '9': 11} | warnings=[]
r3d_18 | 8 FPS | train/val accuracy=0.9960/0.9062 | train/val loss=0.0172/0.3261 | prediction counts={'0': 19, '2': 14, '10': 14, '1': 16, '5': 12, '3': 10, '9': 14, '7': 17, '6': 20, '4': 10, '8': 14} | warnings=[]
r3d_18 | 16 FPS | train/val accuracy=1.0000/0.9625 | train/val loss=0.0002/0.2684 | prediction counts={'0': 19, '4': 13, '1': 11, '5': 18, '2': 13, '3': 14, '6': 18, '7': 17, '8': 14, '9': 12, '10': 11} | warnings=[]
swin3d_t | 4 FPS | train/val accuracy=0.1243/0.1250 | train/val loss=2.3837/2.3842 | prediction counts={'0': 160} | warnings=['single-class predictions', 'at/below training-majority-label validation baseline']
swin3d_t | 8 FPS | train/val accuracy=0.1243/0.1250 | train/val loss=2.3837/2.3843 | prediction counts={'0': 160} | warnings=['single-class predictions', 'at/below training-majority-label validation baseline']
swin3d_t | 16 FPS | train/val accuracy=0.1243/0.1250 | train/val loss=2.3837/2.3842 | prediction counts={'0': 160} | warnings=['single-class predictions', 'at/below training-majority-label validation baseline']
Selected common FPS: 16. Custom-tuned shared FPS, not per-family optimal tuning. Warnings suggest possible optimization failure, not proven capacity limits.
Experiment 128: recipe baseline, LR=0.001, WD=0.0001; test locked
Using local aad dataset: /root/.cache/kagglehub/datasets/sanelehlabisa/abnormal-activities-dataset/versions/2/abnormal-activities-dataset
Device: cuda
Run directory: /content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation/runs/experiments/20261011_140613_467969_abnormal-activities-dataset_recipe-validation
✅ 1069 videos | mode=video | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Loaded 1069 samples | 11 classes: ['Begging', 'Drunkenness', 'Fight', 'Harassment', 'Hijack', 'Knife Hazard', 'Normal Videos', 'Pollution', 'Property Damage', 'Robbery', 'Terrorism']
Train: 748 | Val: 160 | Test locked: 161
Online augmentation: disabled

Experiment 128/142 maximum | final_recipe

Experiment 128/142 maximum | final_recipe: custom_top1 | params=1,077,579
Experiment 128/142 maximum | final_recipe | custom_top1:  25%|██▌       | 64/256 [09:15<27:48,  8.69s/it, lr=1.00e-03, selected=63, train_acc=0.507, train_loss=1.158, val_acc=0.456, val_loss=1.317]
