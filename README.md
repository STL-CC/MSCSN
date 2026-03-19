# MSCSN

Codebase for ECG classification with shapelet-based feature extraction and a downstream neural classifier.

## Repository Scope

- This repository is intended solely for reproducibility purposes during the review process.
- The official source code will be formally released upon the acceptance of the paper.
- Data files are not included. Please prepare data locally following [data/README.md](data/README.md).

## Project Structure

```text
MSCSN/
├── losses/
│   ├── sup_con_loss.py
│   └── sup_info_loss.py
├── model/
│   ├── dnn_classifier.py
│   ├── encoder_with_classifier.py
│   └── trans_tcn_encoder_with_classifier.py
├── data/
│   ├── README.md
│   └── representations/
│       └── <dataset>_train.h5
├── datasets.py
├── pipeline.py
├── train.py
├── utils.py
├── LICENSE
└── README.md
```

## Environment

Python 3.9+ is recommended.

Main dependencies:

- torch
- numpy
- scipy ecosystem tools (scikit-learn)
- wfdb
- pywavelets
- h5py
- tensorboard

Install dependencies manually according to your environment and CUDA setup.

## Data Preparation

1. Read [data/README.md](data/README.md).
2. Download and preprocess datasets into the expected local layout.
3. Place a precomputed representation HDF5 file in `data/representations/`, or allow the pipeline to generate it automatically.

   - `data/representations/MITBIH_train.h5`
   - `data/representations/AFDB_train.h5`

## Training

Run training from repository root:

```bash
python train.py --dataset MITBIH --device cuda --run_stage1
```

Two-stage behavior:

- Stage 1 (`pipeline.py`): checks `data/representations/<dataset>_train.h5`; if missing, trains representation model and exports it.
- Stage 2 (`pipeline.py`): discovers shapelets, extracts features, and trains the classifier.

Use `--run_stage1` to force rebuilding the representation file even if it already exists.

Useful options:

- `--shapelet_num`
- `--hidden_dims`
- `--batch_size`
- `--epochs`
- `--cuml`

Logs and artifacts are saved under `./log/<timestamp>-<dataset>-<model>-<shapelet_num>/`.

## Pipeline Stages

The execution is organized into explicit stages in `pipeline.py`:

1. `stage1_initialize_run`
2. `stage2_load_data`
3. `stage3_prepare_representation_file`
4. `stage4_discover_shapelets`
5. `stage5_extract_features`
6. `stage6_train_classifier`

`train.py` is now a lightweight orchestration entrypoint that preserves the original workflow and core logic.

## License

This project uses a restrictive pre-publication license. See [LICENSE](LICENSE).
