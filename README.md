# ViT Quantization Experiments

Lucrare de licență — **Impactul Cuantizării Numerice asupra Vision Transformers**
Tudose Alexandru · Universitatea Transilvania din Brașov · 2026
Coordonator: Conf. dr. ing. Honorius Gâlmeanu

---

## Descriere

Studiu experimental sistematic al impactului cuantizării numerice (FP16, INT8,
FP8) asupra unui Vision Transformer pretrained (`vit_tiny_patch16_224`) evaluat
pe ImageNet-1k validation (50 000 imagini, 1 000 clase).

---

## Structura proiectului

```
ViT-FP8-experiments/
├── configs/
│   └── preliminary/         # YAML configs pentru studiul CIFAR-10
│
├── src/
│   ├── data/                # Dataset loaders
│   ├── models/
│   │   ├── vit_model.py     # Creare model timm
│   │   └── quantized_linear.py  # INT8 QuantizedLinear (per-tensor + per-channel)
│   ├── training/            # Trainer CIFAR-10 (Faza 0)
│   └── utils/               # Metrici, monitorizare hardware
│
├── scripts/
│   ├── preliminary/         # Faza 0: antrenare CIFAR-10
│   ├── evaluate_fp16_imagenet.py    # Faza 1: FP16 static
│   ├── evaluate_int8_quantization.py # Faza 2: INT8 per-tensor
│   ├── layer_sensitivity_analysis.py # Faza 3: sensitivity analysis
│   ├── generate_fp16_imagenet_plots.py
│   ├── generate_int8_imagenet_plots.py
│   ├── experiments/
│   │   ├── mlp_only_int8.py   # Experiment: cuantizare MLP-only vs full
│   │   └── fp8_vs_int8.py     # Experiment: FP8 E4M3FN vs INT8
│   └── phase4/              # Faza 4: scalare modele mari (Colab)
│       ├── evaluate_model.py
│       ├── evaluate_fp8_native.py
│       ├── compare_cross_model.py
│       └── phase4_colab.ipynb
│
├── docs/
│   ├── raport_studiu_preliminar.tex
│   ├── raport_faza1_fp16.tex
│   ├── raport_faza2_int8.tex
│   ├── raport_faza3_sensitivity.tex
│   ├── raport_experiment_mlp_only.tex
│   ├── raport_experiment_fp8_vs_int8.tex
│   ├── plan_executie_licenta.md
│   └── archive/             # Rapoarte vechi
│
├── results/                 # Rezultate experimente (gitignored)
└── data/                    # Datasets (gitignored)
```

---

## Faze experimentale

### Faza 0 — Studiu Preliminar (CIFAR-10)
Antrenare ViT-Tiny de la zero pe CIFAR-10 cu 4 configurații (FP32/FP16 × Basic/Extended augmentation). Cuantizare post-training FP8 pe cel mai bun model.

```bash
python scripts/preliminary/train.py --config configs/preliminary/AugmFP16.yaml
python scripts/preliminary/evaluate_fp8_quantization.py
```

### Faza 1 — FP16 Static pe ImageNet-1k
Evaluare FP32 baseline vs `model.half()` pe setul de validare ImageNet-1k.

```bash
python scripts/evaluate_fp16_imagenet.py
python scripts/generate_fp16_imagenet_plots.py
```

**Rezultate:** FP32 75.456% → FP16 75.452% (−0.004 pp), 2.00× reducere memorie, 1.22× speedup.

### Faza 2 — INT8 Weight-Only pe ImageNet-1k
Cuantizare INT8 per-tensor cu scalare liniară, selectivă (skip LayerNorm, head).

```bash
python scripts/evaluate_int8_quantization.py
python scripts/generate_int8_imagenet_plots.py
```

**Rezultate:** 75.134% (−0.322 pp), 3.29× reducere memorie.

### Faza 3 — Sensitivity Analysis
Per-tensor vs per-channel, sensitivity per strat (48 evaluări), timing și memory profiling.

```bash
python scripts/layer_sensitivity_analysis.py
# sau fără sensitivity analysis (mai rapid):
python scripts/layer_sensitivity_analysis.py --skip-sensitivity
```

**Rezultate:** INT8-pc 75.424% (−0.032 pp), îmbunătățire MSE 11.14×, outlier block.7.mlp identificat.

### Experimente adiționale

**MLP-only INT8** — cuantizare exclusivă a straturilor MLP, cu straturile de atenție în FP32:
```bash
python scripts/experiments/mlp_only_int8.py
```
**Rezultat:** INT8-mlp-pc 75.486% (+0.030 pp față de FP32), ipoteză confirmată.

**FP8 E4M3FN vs INT8** — replică Faza 2 cu stocare reală în float8_e4m3fn:
```bash
python scripts/experiments/fp8_vs_int8.py
```
**Rezultat:** FP8-pt bate INT8-pt (+0.124 pp), INT8-pc bate FP8-pc (+0.068 pp).

### Faza 4 — Scalare modele mari (Colab Pro+)
ViT-Small (22M) și ViT-Base (86M) pe A100, FP8 hardware nativ.

```bash
# Pe Google Colab Pro+ cu A100:
# Deschide scripts/phase4/phase4_colab.ipynb
```

---

## Rezultate sumar

| Format | Accuracy | Δ FP32 | Memorie | Latență (M4) |
|--------|----------|--------|---------|--------------|
| FP32 | 75.456% | — | 21.81 MB | 200 ms/batch |
| FP16 | 75.452% | −0.004 pp | 10.91 MB | 146 ms/batch |
| INT8-pt | 75.134% | −0.322 pp | 6.62 MB | 200 ms/batch |
| INT8-pc | 75.424% | −0.032 pp | 6.70 MB | 199 ms/batch |
| FP8-pt | 75.258% | −0.198 pp | 6.62 MB | 251 ms/batch |
| FP8-pc | 75.356% | −0.100 pp | 6.70 MB | 275 ms/batch |
| INT8-mlp-pc | 75.486% | +0.030 pp | 11.73 MB | 182 ms/batch |

---

## Setup

```bash
git clone <repository-url>
cd ViT-FP8-experiments
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio timm datasets huggingface_hub \
            pyyaml numpy matplotlib tqdm psutil
```

### Date

ImageNet-1k validation — 14 fișiere parquet în `data/imagenet-1k/`
(descărcare manuală de pe HuggingFace, `data/` este gitignored)

---

## Hardware

Toate experimentele (Fazele 0-3 + experimente adiționale) au rulat pe:
- **MacBook Air, Apple M4, 16 GB RAM**
- Backend PyTorch: MPS (Metal Performance Shaders)
- PyTorch 2.9.1, Python 3.11

Faza 4: Google Colab Pro+, NVIDIA A100 (planificat).

---

## Structura rezultatelor

```
results/
├── preliminary/             # Faza 0: CIFAR-10
├── FP16ImageNet/            # Faza 1
├── INT8ImageNet/            # Faza 2
├── Phase3/                  # Faza 3
├── Phase4/                  # Faza 4 (Colab)
└── experiments/
    ├── mlp_only_int8/
    └── fp8_vs_int8/
```

Toate rezultatele sunt salvate în JSON și gitignored.
