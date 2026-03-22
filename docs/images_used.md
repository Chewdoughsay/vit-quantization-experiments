# Imagini utilizate în raport — Faze 1 și 2

Toate imaginile sunt generate automat de scripturile din `scripts/`.
Căile sunt **relative la fișierul `.tex`** (care se află în `docs/`).

---

## Faza 1 — FP16 Static pe ImageNette

Generat de: `python scripts/generate_fp16_imagenet_plots.py`

| Figura în .tex | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Fig. 1 (summary) | `../results/FP16ImageNet/plots/00_summary.png` | Overview 3 metrici: acuratețe / memorie / latență FP32 vs FP16 |
| Fig. 2a | `../results/FP16ImageNet/plots/01_accuracy_comparison.png` | Bar chart acuratețe FP32 vs FP16 |
| Fig. 2b | `../results/FP16ImageNet/plots/03_latency_comparison.png` | Bar chart latență FP32 vs FP16 |

Imagini disponibile dar **nefolosite direct** în raport (pot fi adăugate):
| Cale | Conținut |
|---|---|
| `../results/FP16ImageNet/plots/02_memory_comparison.png` | Bar chart memorie FP32 vs FP16 |
| `../results/FP16ImageNet/plots/04_accuracy_latency_tradeoff.png` | Scatter accuracy–latency |

---

## Faza 2 — INT8 per-tensor pe ImageNette

Generat de: `python scripts/generate_int8_imagenet_plots.py`

| Figura în .tex | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Fig. 3 (summary) | `../results/INT8ImageNet/plots/00_summary.png` | Overview FP32 vs INT8 (acuratețe / memorie / latență) |
| Fig. 4 (MSE) | `../results/INT8ImageNet/plots/01_per_layer_mse.png` | MSE cuantizare per layer, outlieri marcați |
| Fig. 5 (scale) | `../results/INT8ImageNet/plots/02_per_layer_scale.png` | Distribuție scale factor (scatter + boxplot per tip strat) |
| Fig. 6 (cross) | `../results/INT8ImageNet/plots/03_cross_phase_comparison.png` | Comparație FP32 / FP16 / INT8 side-by-side |
| Fig. 7 (tradeoff) | `../results/INT8ImageNet/plots/04_tradeoff.png` | Scatter accuracy–memorie și accuracy–latență |

---

## Note pentru compilare LaTeX

```latex
% Includere imagine din compilatorul LaTeX (dacă .tex e în docs/):
\includegraphics[width=\textwidth]{../results/FP16ImageNet/plots/00_summary.png}

% Subfiguri:
\begin{subfigure}[b]{0.48\textwidth}
  \includegraphics[width=\textwidth]{../results/FP16ImageNet/plots/01_accuracy_comparison.png}
  \caption{Acuratețe top-1}
\end{subfigure}
```

### Dacă muți `.tex`-ul în altă locație

Ajustează prefixul căii:
- Din `docs/` → `../results/...`
- Din rădăcina proiectului → `results/...`
- Cu cale absolută → `/Users/alextudose/PycharmProjects/ViT-FP8-experiments/results/...`

### Regenerare imagini (dacă rulezi experimente din nou)

```bash
# Faza 1
python scripts/evaluate_fp16_imagenet.py    # → results/FP16ImageNet/metrics/
python scripts/generate_fp16_imagenet_plots.py  # → results/FP16ImageNet/plots/

# Faza 2
python scripts/evaluate_int8_quantization.py    # → results/INT8ImageNet/metrics/
python scripts/generate_int8_imagenet_plots.py  # → results/INT8ImageNet/plots/

# Recompilare raport
cd docs && tectonic raport_faze_1_2.tex
```

---

## Dimensiuni și rezoluții

| Fișier | Dimensiune | DPI (salvat) |
|---|---|---|
| `FP16ImageNet/plots/00_summary.png` | ~1400×500 px | 300 |
| `FP16ImageNet/plots/01_accuracy_comparison.png` | ~600×500 px | 300 |
| `FP16ImageNet/plots/02_memory_comparison.png` | ~600×500 px | 300 |
| `FP16ImageNet/plots/03_latency_comparison.png` | ~600×500 px | 300 |
| `FP16ImageNet/plots/04_accuracy_latency_tradeoff.png` | ~600×500 px | 300 |
| `INT8ImageNet/plots/00_summary.png` | ~1400×500 px | 300 |
| `INT8ImageNet/plots/01_per_layer_mse.png` | ~1600×500 px | 300 |
| `INT8ImageNet/plots/02_per_layer_scale.png` | ~1300×500 px | 300 |
| `INT8ImageNet/plots/03_cross_phase_comparison.png` | ~1500×500 px | 300 |
| `INT8ImageNet/plots/04_tradeoff.png` | ~1300×500 px | 300 |
