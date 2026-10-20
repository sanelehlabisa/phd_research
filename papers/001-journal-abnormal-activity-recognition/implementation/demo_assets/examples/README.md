# Demo examples

Three original AAD version-2 validation clips, copied without editing or
resampling. Selected by class and small file size, not prediction success.
Illustrations only; not an accuracy evaluation. The demo still classifies only
the initial eight frames at 16 FPS, not the entire video.

| Example | Dataset-relative source | Duration | Bytes |
|---|---|---:|---:|
| `normal.mp4` | `Normal Videos/GOPR2234.mp4` | 6.974 s | 25,020 |
| `fight.mp4` | `Fight/GOPR2509_32.mp4` | 6.039 s | 33,131 |
| `pollution.mp4` | `Pollution/GOPR2466_19.mp4` | 2.870 s | 26,301 |

- Source: local Kaggle cache, `sanelehlabisa/abnormal-activities-dataset`, version 2.
- Evidence: `splits/abnormal-activities-dataset_seed42.json`, all three `validation`.
- Manifest SHA256 (LF-normalized): `aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8`, matching demo checkpoint metadata.
- All clips: H.264, 256×256, 29.97 FPS; no augmentation or prediction overlays.
- File SHA256:
  - `normal.mp4`: `a839a0c3a801cddd27bd8cdb12c22f125e0a0bd39dccca9ece506962b1769472`
  - `fight.mp4`: `be2847c0a84c4eea4c59421b068c961ee6a60f30360f4f54e63c0448d6aa41b0`
  - `pollution.mp4`: `dad433a13b3b92560da16e4e4b7a38220b917fe5b3ee08899c7cc1b29d0e81ab`
