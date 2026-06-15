# ViT Quantization Experiments

**Impactul Cuantizării Numerice asupra Vision Transformers**
Tudose Alexandru · Universitatea Transilvania din Brașov · 2026
Coordonator: Conf. dr. ing. Honorius Gâlmeanu

---

## Despre proiect

Acest repository conține codul experimental al lucrării de licență care studiază cum cuantizarea post-training (PTQ) afectează acuratețea unui Vision Transformer pretrained. Experimentele compară formatele FP16, INT8 și FP8 E4M3FN pe setul de validare ImageNet-1k (50.000 de imagini, 1.000 de clase), folosind modelul `vit_tiny_patch16_224` din biblioteca `timm`.

Toate schemele de cuantizare sunt **weight-only**: activările, normalizările și embeddings-urile rămân în float32. Ponderile sunt stocate în formatul țintă și dequantizate la float32 înainte de fiecare înmulțire de matrice — o schemă care aduce exclusiv reducere de memorie, nu speedup de calcul.

Principala concluzie: la granularitate per-channel, INT8 pierde doar −0.032 puncte procentuale față de FP32 pe ViT-Tiny, iar la ViT-Base degradarea scade la −0.002 pp — cuantizarea este practic gratuită pe modele mari.

---

## Setup

```bash
git clone <repository-url>
cd ViT-FP8-experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Date de intrare

Scripturile principale (Fazele 1–4) necesită setul de validare ImageNet-1k în format parquet, descărcat manual de pe HuggingFace și plasat în `data/imagenet-1k/*.parquet`. Directorul `data/` este gitignored. Dacă fișierele parquet nu sunt găsite local, `src/data/imagenet_loader.py` încearcă automat un fallback prin HuggingFace streaming (necesită autentificare).

---

## Structura proiectului

```
ViT-FP8-experiments/
├── configs/
│   ├── FP16ImageNet.yaml
│   ├── INT8ImageNet.yaml
│   └── preliminary/
│       ├── BaseFP32.yaml
│       ├── BaseFP16.yaml
│       ├── AugmFP32.yaml
│       ├── AugmFP16.yaml
│       └── FP8Test.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── imagenet_loader.py
│   ├── models/
│   │   ├── vit_model.py
│   │   └── quantized_linear.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── gpu_monitor.py
│       ├── metrics.py
│       ├── plot_style.py
│       └── system_monitor.py
├── scripts/
│   ├── evaluate_fp16_imagenet.py
│   ├── evaluate_int8_quantization.py
│   ├── layer_sensitivity_analysis.py
│   ├── generate_fp16_imagenet_plots.py
│   ├── generate_int8_imagenet_plots.py
│   ├── analyze_weight_distributions.py
│   ├── preliminary/
│   │   ├── train.py
│   │   ├── evaluate_fp8_quantization.py
│   │   ├── compare_experiments.py
│   │   ├── extract_metrics.py
│   │   ├── generate_plots.py
│   │   ├── generate_report_plots.py
│   │   └── plot_hardware_stats.py
│   ├── experiments/
│   │   ├── fp8_vs_int8.py
│   │   └── mlp_only_int8.py
│   └── phase4/
│       ├── evaluate_model.py
│       ├── evaluate_fp8_native.py
│       ├── compare_cross_model.py
│       └── phase4_colab.ipynb
├── results/            # gitignored — generate local după rulare
├── data/               # gitignored — ImageNet parquet, descărcat manual
└── requirements.txt
```

---

## Modulele din `src/`

### `src/models/quantized_linear.py`

Inima tehnică a proiectului. Implementează cuantizarea INT8 post-training de la zero, fără dependențe externe (fără bitsandbytes sau TensorRT).

Expune primitive pentru quantizare simetrică per-tensor (`int8_quantize`, `int8_dequantize`) și per-channel (`int8_quantize_per_channel`), funcții de eroare MSE/MAE pentru analiză, și două module drop-in pentru `nn.Linear`: `QuantizedLinear` (per-tensor) și `QuantizedLinearPerChannel`. La nivel de model, `quantize_model_selective` și `quantize_model_per_channel` parcurg graful modelului, sar peste straturileLayerNorm, embeddings și capul de clasificare, înlocuiesc în-place restul straturilor `nn.Linear`, și returnează statistici per-strat.

### `src/models/vit_model.py`

Punct de intrare unic pentru crearea oricărui model ViT prin `timm`. Conține și un catalog `MODEL_CONFIGS` cu specificațiile celor trei variante folosite în experimente: ViT-Tiny (5.7M parametri), ViT-Small (22M), ViT-Base (86M).

### `src/evaluation/evaluator.py`

Bucla de evaluare comună tuturor fazelor. Măsoară Top-1 accuracy, cross-entropy loss și latență per-batch (cu warmup și sincronizare MPS/CUDA). Detectează automat modele FP16 și castează intrările corespunzător. Expune și `model_size_mb()` (memorie in-process) și `model_disk_mb()` (dimensiunea `state_dict` serializat).

### `src/data/imagenet_loader.py`

Încarcă setul de validare ImageNet-1k din fișiere parquet locale sau prin HuggingFace streaming, și returnează un `DataLoader` PyTorch standard. Folosit de toate scripturile din Fazele 1–4.

### `src/data/dataset.py`

Data loaders pentru CIFAR-10 cu redimensionare la 224×224 și două niveluri de augmentare (basic și extended). Folosit exclusiv în studiul preliminar (Faza 0); marcat ca legacy.

### `src/training/trainer.py`

Clasa `ViTTrainer` — buclă completă de antrenare cu AMP, gradient clipping, label smoothing, cosine LR scheduling cu warmup, checkpoint-uri pe cel mai bun model și monitorizare hardware per-epocă. Folosit exclusiv pentru antrenarea CIFAR-10 din Faza 0; legacy pentru fazele 1–4.

### `src/utils/plot_style.py`

Configurație globală matplotlib pentru toate figurile din lucrare. Definește fontul (DejaVu Serif), grila, palette de culori colorblind-safe (`COLORS` dict cu chei FP32/FP16/INT8-pt/INT8-pc/FP8), și `save_fig()` care salvează la 300 DPI.

### `src/utils/system_monitor.py` / `gpu_monitor.py` / `metrics.py`

Utilitare de monitorizare hardware și logging de metrici folosite de `ViTTrainer`. `SystemMonitor` urmărește CPU, RAM și throttling termal prin `psutil`. `GPUMonitor` citește puterea GPU Apple Silicon prin `sudo powermetrics`. `MetricsTracker` serializează metricile per-epocă în JSON. Toate sunt legacy — folosite doar în Faza 0.

---

## Scripturile din `scripts/`

### Faza 0 — Studiu preliminar CIFAR-10 (`scripts/preliminary/`)

Antrenează ViT-Tiny de la zero pe CIFAR-10 în patru configurații (FP32/FP16 × Basic/Extended augmentation) și aplică FP8 PTQ pe cel mai bun model ca proof-of-concept al pipeline-ului.

`train.py` — punctul de intrare, citește un fișier YAML și rulează antrenarea completă.
`evaluate_fp8_quantization.py` — aplică FP8 E4M3FN per-tensor și per-channel pe checkpoint-ul AugmFP16 și raportează degradarea pe CIFAR-10.
`compare_experiments.py` — agregă JSON-urile de metrici din toate cele patru experimente și exportă un tabel comparativ.
`extract_metrics.py` — CLI pentru inspectarea individuală a unui JSON de metrici.
`generate_plots.py` — curbe de loss/accuracy per experiment și overlay-uri comparative.
`generate_report_plots.py` — generează cele 9 figuri de publicație ale studiului preliminar.
`plot_hardware_stats.py` — grafice de utilizare CPU/RAM și putere GPU/CPU per experiment.

```bash
python scripts/preliminary/train.py --config configs/preliminary/AugmFP16.yaml
python scripts/preliminary/evaluate_fp8_quantization.py
python scripts/preliminary/generate_report_plots.py
```

### Faza 1 — FP16 pe ImageNet-1k

`scripts/evaluate_fp16_imagenet.py` evaluează FP32 vs `model.half()` pe toate cele 50.000 de imagini și salvează în `results/FP16ImageNet/metrics/fp16_imagenet_results.json`. `scripts/generate_fp16_imagenet_plots.py` citește acel JSON și produce 5 figuri (bare de acuratețe, memorie, latență, summary panel 3-in-1, scatter acuratețe-latență).

```bash
python scripts/evaluate_fp16_imagenet.py
python scripts/generate_fp16_imagenet_plots.py
```

**Rezultate:** FP32 75.456% → FP16 75.452% (−0.004 pp), 2.00× reducere memorie, 1.22× speedup pe M4.

### Faza 2 — INT8 per-tensor pe ImageNet-1k

`scripts/evaluate_int8_quantization.py` aplică cuantizarea INT8 per-tensor selectivă (sare LayerNorm, embeddings, head) și raportează MSE/MAE per strat. `scripts/generate_int8_imagenet_plots.py` produce 5 figuri inclusiv distribuția factorilor de scalare și comparația tripartită FP32/FP16/INT8.

```bash
python scripts/evaluate_int8_quantization.py
python scripts/generate_int8_imagenet_plots.py
```

**Rezultate:** INT8-pt 75.134% (−0.322 pp), 3.29× reducere memorie, fără speedup.

### Faza 3 — Sensitivity analysis și per-channel

`scripts/layer_sensitivity_analysis.py` este scriptul central al Fazei 3. Rulează 7 experimente: baseline FP32, FP16, INT8 per-tensor, INT8 per-channel, comparație MSE per strat, sensitivity analysis (cuantizează câte un strat din 48 izolat, 48 de evaluări complete pe 50k imagini), și profiling de timing/memorie. Produce o heatmap de sensitivitate și alte 4 figuri. Flag-ul `--skip-sensitivity` sare cele 48 de evaluări individuale dacă e nevoie de viteză.

```bash
python scripts/layer_sensitivity_analysis.py
# sau mai rapid:
python scripts/layer_sensitivity_analysis.py --skip-sensitivity
```

**Rezultate:** INT8-pc 75.424% (−0.032 pp), îmbunătățire MSE de 11.14×, outlier identificat: `block.7.mlp.fc1/fc2`.

### Experimente adiționale

`scripts/experiments/mlp_only_int8.py` testează ipoteza că quantizarea exclusivă a straturilor MLP (24 din 48 de straturi, cu atenție în FP32) produce o degradare mai mică. Compară 5 configurații: FP32, INT8-pt full, INT8-pc full, INT8-mlp-pt, INT8-mlp-pc. **Rezultat: INT8-mlp-pc 75.486% (+0.030 pp față de FP32), ipoteză confirmată.**

`scripts/experiments/fp8_vs_int8.py` compară direct FP8 E4M3FN cu INT8 în ambele granularități, cu stocare reală în `torch.float8_e4m3fn`. **Rezultate: FP8-pt bate INT8-pt (+0.124 pp); INT8-pc bate FP8-pc (+0.068 pp).**

```bash
python scripts/experiments/mlp_only_int8.py
python scripts/experiments/fp8_vs_int8.py
```

### Faza 4 — Scalare la modele mari (Google Colab A100)

`scripts/phase4/evaluate_model.py` rulează FP32/FP16/INT8-pt/INT8-pc pentru oricare din cele trei variante ViT (Tiny/Small/Base), auto-selectând CUDA, MPS sau CPU. `scripts/phase4/evaluate_fp8_native.py` adaugă și configurațiile FP8, notând dacă hardware-ul suportă FP8 GEMM nativ (necesită H100; A100 face emulare software). `scripts/phase4/compare_cross_model.py` agregă rezultatele tuturor celor trei modele și generează 7 figuri comparative (scaling de memorie, latență, heatmap de degradare etc.).

`scripts/phase4/phase4_colab.ipynb` este notebook-ul executat pe A100-SXM4-80GB (PyTorch 2.10, CUDA 12.8) care orchestrează tot pipeline-ul Fazei 4, incluzând backup pe Google Drive. Conține outputurile numerice complete ale rulării.

```bash
# Pe Google Colab Pro+ cu A100 — deschide scripts/phase4/phase4_colab.ipynb
```

### `scripts/analyze_weight_distributions.py`

Analizează distribuțiile ponderilor pentru ViT-Tiny/Small/Base (medie, std, skewness, kurtosis per strat) pentru a confirma că straturile `nn.Linear` au ponderi aproape zero-mean — o precondiție pentru cuantizarea simetrică. Salvează statistici JSON și 3 figuri.

---

## Fișierele de configurare (`configs/`)

Fiecare YAML documentează hiper-parametrii unui experiment și înregistrează rezultatele așteptate, servind simultan ca documentație și ca input pentru scripturi.

`FP16ImageNet.yaml` și `INT8ImageNet.yaml` specifică modelul, dimensiunea batch-ului, device-ul și metoda de cuantizare pentru Fazele 1 și 2. YAML-urile din `configs/preliminary/` definesc cele patru configurații de antrenare CIFAR-10 și configurația de evaluare FP8, variind precizia de calcul (FP32 vs FP16 cu AMP) și nivelul de augmentare (basic vs extended).

---

## Rezultate sumar

| Format | Accuracy | Δ FP32 | Memorie | Latență M4 |
|---|---|---|---|---|
| FP32 | 75.456% | — | 21.81 MB | 200 ms/batch |
| FP16 | 75.452% | −0.004 pp | 10.91 MB | 146 ms/batch |
| INT8-pt | 75.134% | −0.322 pp | 6.62 MB | 200 ms/batch |
| INT8-pc | 75.424% | −0.032 pp | 6.70 MB | 199 ms/batch |
| FP8-pt | 75.258% | −0.198 pp | 6.62 MB | 251 ms/batch |
| FP8-pc | 75.356% | −0.100 pp | 6.70 MB | 275 ms/batch |
| INT8-mlp-pc | 75.486% | +0.030 pp | 11.73 MB | 182 ms/batch |

Toate experimentele din Fazele 0–3 au rulat pe MacBook Air cu Apple M4, 16 GB RAM, backend MPS (PyTorch 2.9.1, Python 3.11). Faza 4 a rulat pe Google Colab Pro+ cu NVIDIA A100-SXM4-80GB.

---

## Structura rezultatelor

Rezultatele sunt salvate local în `results/` (gitignored) și organizate astfel:

```
results/
├── preliminary/          # Faza 0: CIFAR-10
├── FP16ImageNet/         # Faza 1
├── INT8ImageNet/         # Faza 2
├── Phase3/               # Faza 3
├── Phase4/               # Faza 4 (Colab)
├── experiments/
│   ├── mlp_only_int8/
│   └── fp8_vs_int8/
└── weight_distributions/
```

Fiecare subdirector conține `metrics/` cu JSON-uri și `plots/` cu figurile generate.
