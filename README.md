# Chord OCR — Open Source

> **Musical chord notation recognition** using ASTER + ConvNeXt architectures,  
> trained on HuggingFace datasets with MLflow tracking.

This project is a clean, open-source implementation of an OCR system specialized in recognizing **musical chord notation** from sheet music images. It uses the ASTER (Attentional Scene Text Recognizer) attention-based architecture combined with a ConvNeXt feature extractor, a fully YAML-driven configuration system, and HuggingFace Datasets as the data source.

Built as a production-grade ML training pipeline with:
- 🤗 HuggingFace Datasets as the only data source (no local LMDB required)
- 🧠 ASTER attention decoder + ConvNeXt backbone
- ⚙️ Full YAML config system for training and datasets
- 📊 MLflow dashboard for experiment tracking
- 🏫 Curriculum learning support
- 🔄 Streaming publisher with augmentation pipeline
- 🎯 Specialized validation metrics for chord notation

---

## Architecture Overview

```
HuggingFace Datasets (array)
        ↓
  HF Publisher (streaming, augmentation, curriculum)
        ↓
  Training Loop (train.py)
        ↓
  ┌─────────────────────────────────────────┐
  │  ASTER / ConvNeXt OCR Model             │
  │  ┌──────────┐  ┌────────┐  ┌─────────┐ │
  │  │ ConvNeXt │→ │ BiLSTM │→ │ ASTER   │ │
  │  │ Backbone │  │Encoder │  │Attention│ │
  │  └──────────┘  └────────┘  └─────────┘ │
  └─────────────────────────────────────────┘
        ↓
  Validation + MLflow Tracking
```

---

## Implementation Roadmap

> **Legend:** ✅ Done · 🔄 In Progress · ⬜ Pending

---

### PHASE 0 — Repository Bootstrap

- [ ] **0.1** Create repository structure (folders, `__init__.py`, `.gitignore`, `LICENSE`)
- [ ] **0.2** Write `requirements.txt` (torch, transformers, datasets, mlflow, pyyaml, pydantic, albumentations, Pillow, editdistance)
- [ ] **0.3** Write `pyproject.toml` / `setup.py` for pip-installable package
- [ ] **0.4** Add `.env.example` documenting all environment variables
- [ ] **0.5** Set up `pre-commit` hooks (black, isort, mypy)
- [ ] **0.6** Add GitHub Actions CI workflow (lint + unit tests)
- [ ] **0.7** Add as git submodule in parent OCR-Chord-Notation project

---

### PHASE 1 — Core Architecture Extraction

> Extract ASTER + ConvNeXt from private repo. Clean, no TrOCR, no binary classifier, no R&D mess.

#### 1.1 — ConvNeXt Backbone
- [ ] **1.1.1** Copy `ocr/architecture/convnext/model.py` → `chord_ocr/architecture/convnext/model.py`
- [ ] **1.1.2** Copy `ocr/architecture/convnext/feature_extractor.py` → `chord_ocr/architecture/convnext/feature_extractor.py`
- [ ] **1.1.3** Remove any LMDB/Redis/private imports, make self-contained
- [ ] **1.1.4** Write unit test: `tests/test_convnext.py` — forward pass with dummy tensor `(B, 1, 96, 192)`
- [ ] **1.1.5** Document: ConvNeXt variant used, layer config, output shape

#### 1.2 — ASTER Attention Architecture
- [ ] **1.2.1** Copy `ocr/architecture/aster/model.py` → `chord_ocr/architecture/aster/model.py`
- [ ] **1.2.2** Copy `ocr/architecture/aster/encoder.py` → BiLSTM encoder
- [ ] **1.2.3** Copy `ocr/architecture/aster/decoder.py` → Attention decoder
- [ ] **1.2.4** Copy `ocr/architecture/aster/attention.py` → Attention mechanism
- [ ] **1.2.5** Write unit test: `tests/test_aster.py` — end-to-end forward pass
- [ ] **1.2.6** Document: ASTER architecture diagram, input/output specs

#### 1.3 — ResNet Backbone (alternative to ConvNeXt)
- [ ] **1.3.1** Copy `ocr/architecture/feature_extraction.py` → `chord_ocr/architecture/feature_extraction.py` (ResNet only)
- [ ] **1.3.2** Write unit test for ResNet feature extractor

#### 1.4 — BiLSTM Sequence Modeling
- [ ] **1.4.1** Copy `ocr/architecture/lstm/sequence_modeling.py` → `chord_ocr/architecture/sequence_modeling.py`
- [ ] **1.4.2** Write unit test

#### 1.5 — Unified Model Entry Point
- [ ] **1.5.1** Adapt `ocr/architecture/model.py` → `chord_ocr/architecture/model.py`
  - Keep: ConvNeXt + ResNet feature extractors
  - Keep: BiLSTM sequence modeling
  - Keep: ASTER attention prediction + CTC prediction
  - Remove: TrOCR, experimental models
- [ ] **1.5.2** Write integration test: build model from YAML config, run forward pass
- [ ] **1.5.3** Add `ModelFactory.from_config(opt)` class method

---

### PHASE 2 — YAML Configuration System

> Fully declarative training via YAML. One config = one experiment.

#### 2.1 — DTO / Pydantic Schema
- [ ] **2.1.1** Port `dto.py` → `chord_ocr/config/schema.py`
  - `TrainingConfig` — all training hyperparameters
  - `DatasetConfig` — HuggingFace dataset definition (see Phase 3)
  - `PhaseConfig` — curriculum learning phase
  - `MLflowConfig` — experiment tracking settings (new)
  - `AugmentationConfig` — augmentation pipeline settings
- [ ] **2.1.2** Add strict validation: unknown fields raise error, required fields documented
- [ ] **2.1.3** Write unit tests for config parsing edge cases

#### 2.2 — Config Loader
- [ ] **2.2.1** Port `get_config.py` → `chord_ocr/config/loader.py`
- [ ] **2.2.2** Support config inheritance: `base_config: path/to/base.yaml`
- [ ] **2.2.3** Support env variable substitution: `train_data: ${HF_DATASET_NAME}`

#### 2.3 — Example Configs
- [ ] **2.3.1** `configs/training/aster_convnext_base.yaml` — baseline ASTER + ConvNeXt
- [ ] **2.3.2** `configs/training/aster_resnet_base.yaml` — ASTER + ResNet (lightweight)
- [ ] **2.3.3** `configs/training/curriculum_example.yaml` — multi-phase curriculum learning
- [ ] **2.3.4** `configs/datasets/huggingface_example.yaml` — HuggingFace dataset config
- [ ] **2.3.5** `configs/datasets/multi_dataset_example.yaml` — array of HF datasets with weights

---

### PHASE 3 — HuggingFace Dataset Integration

> Replace LMDB + Redis publisher. New publisher reads from HF Datasets array.

#### 3.1 — HuggingFace Dataset Config Schema
- [ ] **3.1.1** Define `HFDatasetConfig` in schema:
  ```yaml
  datasets:
    - repo_id: "charlyjaquez/chord-notation-synthetic"
      split: "train"
      weight: 0.7
      label_column: "label"
      image_column: "image"
    - repo_id: "charlyjaquez/chord-notation-real"
      split: "train"
      weight: 0.3
      label_column: "chord"
      image_column: "img"
  ```
- [ ] **3.1.2** Support: streaming mode (`streaming: true`) to avoid full download
- [ ] **3.1.3** Support: local cache path (`cache_dir: D:/hf_cache`)
- [ ] **3.1.4** Support: dataset filters (`filter_fn: "lambda x: len(x['label']) <= 12"`)

#### 3.2 — HuggingFace Multi-Dataset Publisher
- [ ] **3.2.1** Create `chord_ocr/data/publisher/hf_publisher.py`
  - Loads N HuggingFace datasets from config array
  - Weighted sampling across datasets
  - Streams batches with augmentation pipeline
  - Curriculum-aware: filters by chord length per phase
- [ ] **3.2.2** Create `chord_ocr/data/publisher/image_processing.py`
  - Port augmentation pipeline from private repo
  - Albumentations-based: blur, perspective, noise, rotation, thresholding
  - Music-sheet specific noise patterns
- [ ] **3.2.3** Create `chord_ocr/data/publisher/curriculum_learning.py`
  - Phase-based chord length filtering
  - Per-phase distribution targets
- [ ] **3.2.4** Create `chord_ocr/data/publisher/thread_workers.py`
  - Multi-threaded batch prefetching
  - Queue-based producer/consumer pattern

#### 3.3 — Dataset Consumer (PyTorch DataLoader compatible)
- [ ] **3.3.1** Create `chord_ocr/data/hf_dataset.py`
  - `HFChordDataset(Dataset)` — wraps HF dataset for PyTorch
  - Handles image resize, grayscale, normalization
  - Compatible with `AlignCollate` batch collator
- [ ] **3.3.2** Create `chord_ocr/data/align_collate.py`
  - Port from private repo
  - Pad images to uniform size within batch
  - Contrast adjustment support
- [ ] **3.3.3** Create `chord_ocr/data/sampler.py`
  - `CurriculumSampler` — curriculum-aware weighted sampling
  - Works with multi-dataset weighted mixing

#### 3.4 — Dataset YAML Config for HuggingFace Generator (synthetic data)
- [ ] **3.4.1** Create `chord_ocr/data/generator/hf_generator_config.py`
  - YAML schema for synthetic chord image generation
  - Font list, background types, augmentation levels
  - Upload target: HuggingFace Hub repo
- [ ] **3.4.2** Create `chord_ocr/data/generator/hf_uploader.py`
  - Generate synthetic images → upload directly to HF Hub
  - Incremental upload (resumable)
- [ ] **3.4.3** `configs/generators/synthetic_base.yaml` — example generator config

#### 3.5 — Tests
- [ ] **3.5.1** `tests/test_hf_publisher.py` — mock HF dataset, test sampling
- [ ] **3.5.2** `tests/test_curriculum_sampler.py` — verify phase transitions
- [ ] **3.5.3** `tests/test_align_collate.py` — batch collation shapes

---

### PHASE 4 — Training Pipeline

#### 4.1 — Label Converters
- [ ] **4.1.1** Port `ocr/train/utils.py` → `chord_ocr/train/utils.py`
  - `CTCLabelConverter` — encode/decode for CTC prediction
  - `AttnLabelConverter` — encode/decode for ASTER attention
  - `Averager` — running average for metrics

#### 4.2 — Forward Pass
- [ ] **4.2.1** Port `ocr/train/forward_pass.py` → `chord_ocr/train/forward_pass.py`
  - Mixed precision (`torch.amp.autocast`)
  - CTC loss + Attention cross-entropy loss
  - Gradient clipping

#### 4.3 — Main Training Loop
- [ ] **4.3.1** Port `ocr/train/train.py` → `chord_ocr/train/train.py`
  - Replace LMDB/Redis DataLoader → HF DataLoader
  - Keep: curriculum phase management
  - Keep: checkpoint save/load (`save_every_n_iterations`)
  - Keep: best model tracking (best_accuracy, best_norm_ed)
  - Add: MLflow logging at every `val_interval`
  - Add: HuggingFace Hub model upload on new best (optional)
- [ ] **4.3.2** Port `run.py` → `chord_ocr/train/run.py` (entry point)
- [ ] **4.3.3** Add `chord_ocr/train/resume.py` — clean checkpoint resume logic

#### 4.4 — Gradient Health Monitor
- [ ] **4.4.1** Port `ocr/train/log_gradient_health.py` → `chord_ocr/train/gradient_monitor.py`
  - Log gradient norms per layer
  - Detect vanishing/exploding gradients
  - Write to MLflow as metrics

#### 4.5 — Tests
- [ ] **4.5.1** `tests/test_forward_pass.py` — CTC and Attn forward with dummy data
- [ ] **4.5.2** `tests/test_label_converters.py` — encode/decode round-trip

---

### PHASE 5 — Validation & Metrics

#### 5.1 — Core Validation Loop
- [ ] **5.1.1** Port `ocr/train/validation.py` → `chord_ocr/train/validation.py`
  - EOS token tracking
  - Edit distance metrics
  - Per-sample prediction table (ground truth vs prediction)

#### 5.2 — Extended Chord Metrics
- [ ] **5.2.1** Port `ocr/train/validation_new_metrics.py` → `chord_ocr/train/chord_metrics.py`
  - `CharacterErrorRate (CER)`
  - `AccuracyByChordLength` — groups: 1-6, 7-12, 13-18, 19+
  - `AccuracyByChordType` — Simple / Seventh / Extended / Altered / Slash / Complex
  - `ConfidenceCalibration` — gap between correct vs incorrect confidence
  - `TopKCharacterConfusions` — most common misrecognized characters
- [ ] **5.2.2** Write unit tests for each metric class

#### 5.3 — Validation Log Writer
- [ ] **5.3.1** Create `chord_ocr/train/validation_logger.py`
  - Structured validation report (text + JSON)
  - Write to file AND MLflow artifacts
  - Format: iteration, CER, accuracy by type, top confusions

#### 5.4 — Tests
- [ ] **5.4.1** `tests/test_chord_metrics.py` — all metric classes with known inputs/outputs

---

### PHASE 6 — MLflow Integration

> Replace file-based CSV/txt logging with MLflow experiment tracking.

#### 6.1 — MLflow Tracker
- [ ] **6.1.1** Create `chord_ocr/monitoring/mlflow_tracker.py`
  - `ExperimentTracker` class wrapping MLflow
  - `log_training_step(iter, loss, grad_norm, lr)`
  - `log_validation(iter, cer, accuracy_by_type, norm_ed)`
  - `log_model_checkpoint(path, metrics)`
  - `log_config(opt)` — log full YAML config as artifact
  - `log_sample_predictions(table)` — log prediction table as artifact

#### 6.2 — MLflow Config in YAML
- [ ] **6.2.1** Add `MLflowConfig` to schema:
  ```yaml
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"   # or "mlruns/" for local
    experiment_name: "chord-ocr-aster-convnext"
    run_name: "phase4-500k"
    tags:
      architecture: "aster+convnext"
      dataset: "charlyjaquez/chord-notation-synthetic"
  ```
- [ ] **6.2.2** Support remote tracking server (MLflow + S3/GCS artifact store)
- [ ] **6.2.3** `configs/mlflow/local.yaml` — local SQLite backend
- [ ] **6.2.4** `configs/mlflow/remote.yaml` — remote server template

#### 6.3 — MLflow Dashboard Setup
- [ ] **6.3.1** Create `scripts/start_mlflow.sh` — launch MLflow UI
- [ ] **6.3.2** Create `scripts/start_mlflow.bat` — Windows version
- [ ] **6.3.3** Create `docker/mlflow/docker-compose.yml` — MLflow + PostgreSQL + MinIO (S3-compatible artifact store)
- [ ] **6.3.4** Document dashboard: which metrics to watch, how to compare runs
- [ ] **6.3.5** Add `docs/mlflow_guide.md` — screenshots and walkthrough

#### 6.4 — Experiment Comparison Utilities
- [ ] **6.4.1** Create `chord_ocr/monitoring/compare_runs.py`
  - Query MLflow API, compare runs by metric
  - Plot accuracy progression across experiments
  - Export comparison CSV

---

### PHASE 7 — Augmentation Pipeline

#### 7.1 — Core Augmentations
- [ ] **7.1.1** Port `ocr/dataset/augmentations/` → `chord_ocr/data/augmentations/`
  - `pipeline.py` — sequential augmentation builder from YAML
  - `rotation.py` — custom rotation
  - `music_sheet_noise.py` — music-specific noise patterns
  - `random_padding.py` — random padding
  - `thresholding.py` — adaptive thresholding

#### 7.2 — Augmentation Config in YAML
- [ ] **7.2.1** Global augmentation config:
  ```yaml
  augmentation:
    enabled: true
    level: "medium"   # low / medium / high
    blur_probability: 0.3
    perspective_probability: 0.2
    noise_probability: 0.4
    music_sheet_noise: true
  ```
- [ ] **7.2.2** Per-phase augmentation override in curriculum config
- [ ] **7.2.3** Write unit tests: augmentation doesn't crash on edge-case images

---

### PHASE 8 — Inference & API

#### 8.1 — Predictor Class
- [ ] **8.1.1** Create `chord_ocr/inference/predictor.py`
  - `ChordOCRPredictor.from_checkpoint(path, config)`
  - `predict(image: PIL.Image) → str`
  - `predict_batch(images: List[PIL.Image]) → List[str]`
  - Returns confidence scores alongside predictions

#### 8.2 — HuggingFace Hub Model Export
- [ ] **8.2.1** Create `chord_ocr/inference/hf_export.py`
  - Export best checkpoint to HuggingFace Hub
  - Include model card with training config and metrics
- [ ] **8.2.2** `configs/export/hf_hub.yaml` — Hub repo config

#### 8.3 — Simple FastAPI Server
- [ ] **8.3.1** Create `chord_ocr/api/server.py`
  - `POST /predict` — image → chord string
  - `GET /health` — model status
  - `GET /metrics` — last validation metrics
- [ ] **8.3.2** `scripts/start_api.sh` / `.bat`
- [ ] **8.3.3** `docker/api/Dockerfile`

---

### PHASE 9 — Documentation & Examples

#### 9.1 — Main Documentation
- [ ] **9.1.1** `docs/architecture.md` — ASTER + ConvNeXt deep dive with diagrams
- [ ] **9.1.2** `docs/training_guide.md` — step-by-step training walkthrough
- [ ] **9.1.3** `docs/dataset_guide.md` — how to create and upload HF datasets
- [ ] **9.1.4** `docs/curriculum_learning.md` — curriculum phases explained
- [ ] **9.1.5** `docs/mlflow_guide.md` — MLflow dashboard walkthrough
- [ ] **9.1.6** `docs/config_reference.md` — all YAML fields documented with types and defaults

#### 9.2 — Jupyter Notebooks (Portfolio Showcase)
- [ ] **9.2.1** `notebooks/01_data_exploration.ipynb` — explore HF dataset, distribution analysis
- [ ] **9.2.2** `notebooks/02_architecture_walkthrough.ipynb` — ASTER + ConvNeXt forward pass explained
- [ ] **9.2.3** `notebooks/03_training_demo.ipynb` — mini training run on small HF subset
- [ ] **9.2.4** `notebooks/04_mlflow_analysis.ipynb` — analyze training results from MLflow
- [ ] **9.2.5** `notebooks/05_error_analysis.ipynb` — analyze prediction errors and confusions

#### 9.3 — Examples
- [ ] **9.3.1** `examples/quick_start.py` — 10-line training example
- [ ] **9.3.2** `examples/predict_image.py` — load checkpoint, predict single image
- [ ] **9.3.3** `examples/export_to_hf.py` — export trained model to HF Hub

---

### PHASE 10 — CI/CD & Release

- [ ] **10.1** GitHub Actions: `test.yml` — run pytest on push
- [ ] **10.2** GitHub Actions: `lint.yml` — black + isort + mypy
- [ ] **10.3** GitHub Actions: `publish_model.yml` — on tag, export best model to HF Hub
- [ ] **10.4** `CONTRIBUTING.md` — contribution guidelines
- [ ] **10.5** `CHANGELOG.md` — version history
- [ ] **10.6** `v0.1.0` release tag — baseline architecture extracted and working
- [ ] **10.7** `v0.2.0` release tag — HuggingFace integration complete
- [ ] **10.8** `v1.0.0` release tag — full pipeline + MLflow + API working

---

## Repository Structure (Target)

```
chord-ocr-open/
├── chord_ocr/
│   ├── architecture/
│   │   ├── model.py              # Unified model factory
│   │   ├── feature_extraction.py # ResNet backbone
│   │   ├── sequence_modeling.py  # BiLSTM
│   │   ├── convnext/
│   │   │   ├── model.py
│   │   │   └── feature_extractor.py
│   │   └── aster/
│   │       ├── model.py
│   │       ├── encoder.py
│   │       ├── decoder.py
│   │       └── attention.py
│   ├── config/
│   │   ├── schema.py             # Pydantic DTOs
│   │   └── loader.py             # YAML loader
│   ├── data/
│   │   ├── hf_dataset.py         # HuggingFace PyTorch Dataset
│   │   ├── align_collate.py      # Batch collation
│   │   ├── sampler.py            # Curriculum sampler
│   │   ├── augmentations/        # Augmentation pipeline
│   │   └── publisher/
│   │       ├── hf_publisher.py   # HF multi-dataset publisher
│   │       ├── image_processing.py
│   │       ├── curriculum_learning.py
│   │       └── thread_workers.py
│   ├── train/
│   │   ├── run.py                # CLI entry point
│   │   ├── train.py              # Training loop
│   │   ├── forward_pass.py       # Forward pass + loss
│   │   ├── validation.py         # Validation loop
│   │   ├── chord_metrics.py      # Chord-specific metrics
│   │   ├── validation_logger.py  # Structured validation output
│   │   ├── gradient_monitor.py   # Gradient health
│   │   └── utils.py              # Label converters
│   ├── monitoring/
│   │   ├── mlflow_tracker.py     # MLflow integration
│   │   └── compare_runs.py       # Run comparison utilities
│   ├── inference/
│   │   ├── predictor.py          # ChordOCRPredictor
│   │   └── hf_export.py          # HF Hub exporter
│   └── api/
│       └── server.py             # FastAPI server
├── configs/
│   ├── training/
│   │   ├── aster_convnext_base.yaml
│   │   ├── aster_resnet_base.yaml
│   │   └── curriculum_example.yaml
│   ├── datasets/
│   │   ├── huggingface_example.yaml
│   │   └── multi_dataset_example.yaml
│   ├── generators/
│   │   └── synthetic_base.yaml
│   └── mlflow/
│       ├── local.yaml
│       └── remote.yaml
├── docs/
│   ├── architecture.md
│   ├── training_guide.md
│   ├── dataset_guide.md
│   ├── curriculum_learning.md
│   ├── mlflow_guide.md
│   └── config_reference.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_architecture_walkthrough.ipynb
│   ├── 03_training_demo.ipynb
│   ├── 04_mlflow_analysis.ipynb
│   └── 05_error_analysis.ipynb
├── examples/
│   ├── quick_start.py
│   ├── predict_image.py
│   └── export_to_hf.py
├── tests/
│   ├── test_convnext.py
│   ├── test_aster.py
│   ├── test_hf_publisher.py
│   ├── test_curriculum_sampler.py
│   ├── test_align_collate.py
│   ├── test_forward_pass.py
│   ├── test_label_converters.py
│   └── test_chord_metrics.py
├── scripts/
│   ├── start_mlflow.sh
│   ├── start_mlflow.bat
│   └── start_api.sh
├── docker/
│   ├── mlflow/
│   │   └── docker-compose.yml
│   └── api/
│       └── Dockerfile
├── .github/
│   └── workflows/
│       ├── test.yml
│       ├── lint.yml
│       └── publish_model.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

---

## Quick Start (Target — after v1.0.0)

```bash
git clone https://github.com/YOUR_USERNAME/chord-ocr-open
cd chord-ocr-open
pip install -r requirements.txt

# Train with HuggingFace dataset
python -m chord_ocr.train.run --config configs/training/aster_convnext_base.yaml

# Launch MLflow dashboard
bash scripts/start_mlflow.sh
# → open http://localhost:5000

# Predict a chord image
python examples/predict_image.py --image path/to/chord.png --checkpoint checkpoints/best.pth
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.x |
| Architectures | ASTER (custom), ConvNeXt (custom, no TIMM) |
| Data | HuggingFace Datasets + `datasets` library |
| Augmentation | Albumentations + custom music-sheet noise |
| Config | YAML + Pydantic v2 |
| Experiment Tracking | MLflow |
| API | FastAPI |
| CI/CD | GitHub Actions |
| Packaging | `pyproject.toml` |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*This project is part of a portfolio demonstrating production-grade ML engineering for musical score analysis.*
