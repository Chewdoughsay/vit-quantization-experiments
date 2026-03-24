# Imagini utilizate în rapoarte

Toate imaginile sunt generate automat de scripturile din `scripts/`.
Căile sunt **relative la fișierul `.tex`** (care se află în `docs/`).

---

## Studiu Preliminar — CIFAR-10 (Faza 0)

Generat de: `python scripts/preliminary/generate_report_plots.py`

### Folosite în `raport_studiu_preliminar.tex`

| Figura | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Fig. 1 (val acc) | `../results/preliminary/report_plots/A1_val_acc_all.png` | Curbe validare — 4 experimente suprapuse |
| Fig. 2 (grid) | `../results/preliminary/report_plots/A2_train_val_acc_grid.png` | Grid 2x2 train vs val accuracy + overfit shading |
| Fig. 3a (acc bar) | `../results/preliminary/report_plots/C1_best_val_acc_bar.png` | Bar chart acuratețe maximă |
| Fig. 3b (overfit) | `../results/preliminary/report_plots/C2_overfitting_gap_bar.png` | Bar chart overfitting gap |
| Fig. 4 (loss) | `../results/preliminary/report_plots/B_loss_curves.png` | Curbe loss (train + val) |
| Fig. 5 (time) | `../results/preliminary/report_plots/C3_training_time_bar.png` | Bar chart timp antrenare |
| Fig. 6 (FP8) | `../results/preliminary/report_plots/D_fp8_impact.png` | Impact cuantizare FP8 per-tensor vs per-channel |
| Fig. 7a (hw) | `../results/preliminary/report_plots/E1_cpu_memory_bar.png` | CPU și RAM per experiment |
| Fig. 7b (power) | `../results/preliminary/report_plots/E2_gpu_power_profile.png` | Profil putere GPU/CPU |

### Per-experiment (disponibile, nefolosite direct)

| Cale | Conținut |
|---|---|
| `../results/preliminary/*/plots/*_training_curves.png` | Curbe train/val per experiment |
| `../results/preliminary/*/plots/*_learning_rate.png` | LR schedule per experiment |
| `../results/preliminary/*/plots/*_hardware_monitoring.png` | CPU/RAM/thermal time series |
| `../results/preliminary/*/plots/*_power_consumption.png` | Profil putere (doar BaseFP32, AugmFP32) |
| `../results/preliminary/comparison_plots/comparison_*.png` | Overlay train/val acc/loss |

---

## Faza 1 — FP16 Static pe ImageNette

Generat de: `python scripts/generate_fp16_imagenet_plots.py`

### Folosite în `raport_faze_1_2_3.tex`

| Figura | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Summary FP16 | `../results/FP16ImageNet/plots/00_summary.png` | Overview 3 metrici: acuratețe / memorie / latență FP32 vs FP16 |
| Acc comparison | `../results/FP16ImageNet/plots/01_accuracy_comparison.png` | Bar chart acuratețe FP32 vs FP16 |
| Latency | `../results/FP16ImageNet/plots/03_latency_comparison.png` | Bar chart latență FP32 vs FP16 |

### Disponibile dar nefolosite direct

| Cale | Conținut |
|---|---|
| `../results/FP16ImageNet/plots/02_memory_comparison.png` | Bar chart memorie FP32 vs FP16 |
| `../results/FP16ImageNet/plots/04_accuracy_latency_tradeoff.png` | Scatter accuracy–latency |

---

## Faza 2 — INT8 per-tensor pe ImageNette

Generat de: `python scripts/generate_int8_imagenet_plots.py`

### Folosite în `raport_faze_1_2_3.tex`

| Figura | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Summary INT8 | `../results/INT8ImageNet/plots/00_summary.png` | Overview FP32 vs INT8 (acuratețe / memorie / latență) |
| MSE per layer | `../results/INT8ImageNet/plots/01_per_layer_mse.png` | MSE cuantizare per layer, outlieri marcați |
| Scale factors | `../results/INT8ImageNet/plots/02_per_layer_scale.png` | Distribuție scale factor (scatter + boxplot per tip strat) |
| Cross-phase | `../results/INT8ImageNet/plots/03_cross_phase_comparison.png` | Comparație FP32 / FP16 / INT8 side-by-side |
| Tradeoff | `../results/INT8ImageNet/plots/04_tradeoff.png` | Scatter accuracy–memorie și accuracy–latență |

---

## Faza 3 — Layer Sensitivity Analysis

Generat de: `python scripts/layer_sensitivity_analysis.py`

### Folosite în `raport_faze_1_2_3.tex`

| Figura | Cale relativă din `docs/` | Conținut |
|---|---|---|
| Global comparison | `../results/Phase3/plots/00_global_comparison.png` | Bar chart FP32/FP16/INT8-pt/INT8-pc: acc + mem + latency |
| Sensitivity ranked | `../results/Phase3/plots/01_sensitivity_ranked.png` | Bar chart degradare per layer (sortat), 48 layers |
| MSE pt vs pc | `../results/Phase3/plots/02_mse_per_tensor_vs_per_channel.png` | MSE per-tensor vs per-channel per layer + improvement factor |
| Timing + memory | `../results/Phase3/plots/03_timing_memory.png` | Bar chart latency + memory footprint per format |
| Sensitivity heatmap | `../results/Phase3/plots/04_sensitivity_heatmap.png` | Heatmap 12 blocks x 4 layer types, culoare = degradare |

---

## Regenerare imagini

```bash
# Studiu preliminar (Faza 0)
python scripts/preliminary/generate_report_plots.py

# Faza 1
python scripts/evaluate_fp16_imagenet.py
python scripts/generate_fp16_imagenet_plots.py

# Faza 2
python scripts/evaluate_int8_quantization.py
python scripts/generate_int8_imagenet_plots.py

# Faza 3
python scripts/layer_sensitivity_analysis.py

# Compilare rapoarte
cd docs && tectonic raport_studiu_preliminar.tex
cd docs && tectonic raport_faze_1_2_3.tex
```
