# Plan de Execuție — Lucrare de Licență
## Impactul Cuantizării Numerice asupra Vision Transformers
### Tudose Alexandru · Coordonator: Conf. dr. ing. Honorius Gâlmeanu

---

## Context

Studiul preliminar (Faza 0) este **finalizat**: ViT-Tiny pe CIFAR-10, antrenat în FP32/FP16, cuantizat post-training la FP8 E4M3FN. Rezultate cheie: degradare 0.01–0.02 pp, viabilitate demonstrată.

Pe baza discuției cu coordonatorul, direcția se schimbă de la "FLUX.1 pe NVIDIA" la o **cuantizare riguroasă, pas cu pas**, cu corecții tehnice importante.

---

## Corecții critice de la coordonator

### 1. Metoda FP8 curentă — problematică

Implementarea actuală face `cast` direct la `torch.float8_e4m3fn`:

```python
# PROBLEMATIC — nu toate valorile FP32 sunt reprezentabile în FP8
quantized = (cpu * scale.cpu()).clamp(-FP8_MAX, FP8_MAX)
quantized = quantized.to(torch.float8_e4m3fn).to(cpu.dtype)
```

**Corectă** este abordarea **INT8 cu scalare liniară**:

```python
def fp8_quantize(tensor):
    """Cuantizare corectă: scalare liniară + conversie la int8"""
    scale = tensor.abs().max() / 127.0
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale

def fp8_dequantize(q, scale):
    """Decuantizare: int8 → float"""
    return q.float() * scale
```

Aceasta garantează o mapare bijectivă la un set discret de 256 valori, spre deosebire de FP8 nativ unde nu toate valorile intermediare sunt reprezentabile.

### 2. LayerNorm / BatchNorm — NU se cuantizează

Straturile de normalizare sunt sensibile la erori de cuantizare (operează pe statistici de medie/varianță). Ele rămân **obligatoriu în FP32** (sau FP16). Cuantizarea se aplică doar pe:

- `nn.Linear` (proiecțiile Q, K, V, out din attention)
- `nn.Linear` (layerele FFN / MLP)
- Patch embedding (`nn.Conv2d` sau `nn.Linear`)

### 3. Dataset — ImageNet pretrained, nu CIFAR-10

CIFAR-10 este prea mic și simplu. Modelul ViT-Tiny **pretrained pe ImageNet** de la `timm` se încarcă cu `pretrained=True` și se evaluează direct — fără antrenare.

Experimentele pe CIFAR-10 se păstrează ca **test funcțional** (proof of concept).

---

## Faza 1 — Cuantizare FP16 statică pe ViT-Tiny pretrained

**Obiectiv:** Verificarea că inferența pe MacBook funcționează corect cu model pretrained pe ImageNet, cuantizat la FP16 (half precision).

**Pași:**

1. Încarcă `vit_tiny_patch16_224` cu `pretrained=True` din `timm`
2. Descarcă un subset de validare ImageNet (sau ImageNet-V2 / ImageNette ca proxy)
3. Evaluează modelul original (FP32) — notează top-1 accuracy de referință
4. Convertește toate ponderile la FP16 (`model.half()`)
5. Evaluează modelul FP16 pe aceleași date
6. Compară: acuratețe, latență per batch, memorie

**Livrabile:** Script `evaluate_fp16_imagenet.py`, rezultate JSON, comparație FP32 vs FP16.

**Hardware:** MacBook Air M4, 16GB RAM, MPS backend.

**Notă:** Aceasta este o cuantizare **statică** (nu antrenare) — pur inferență.

---

## Faza 2 — Cuantizare FP8 via INT8 scaling (doar MatMul)

**Obiectiv:** Implementare corectă a cuantizării pe 8 biți cu scalare liniară, aplicată selectiv doar pe straturile de tip matmul.

**Pași:**

1. Implementează funcții `int8_quantize` / `int8_dequantize` cu scalare liniară
2. Iterează prin model, identifică straturile cuantizabile:
   - `nn.Linear` în blocurile Attention (qkv, proj)
   - `nn.Linear` în blocurile MLP (fc1, fc2)
   - Patch embedding (dacă e Linear)
3. **NU cuantiza:** LayerNorm, cls_token, pos_embed, head (classification head)
4. Per fiecare strat cuantizabil: salvează `(q_int8, scale)`, la inferență decuantizează
5. Evaluează pe ImageNet validation subset
6. Compară cu FP32 baseline: acuratețe, degradare per-layer

**Implementare sugerată:**

```python
def quantize_model_selective(model):
    """Cuantizare selectivă: doar Linear din Attention + MLP"""
    skip_patterns = ['norm', 'cls_token', 'pos_embed', 'head']
    
    for name, module in model.named_modules():
        if any(p in name for p in skip_patterns):
            continue
        if isinstance(module, nn.Linear):
            q, scale = int8_quantize(module.weight.data)
            # Înlocuiește cu wrapper cuantizat
            ...
```

**Livrabile:** Script `evaluate_int8_quantization.py`, wrapper `QuantizedLinear`, rezultate JSON.

---

## Faza 3 — Extindere și comparații

**Obiectiv:** Analiză sistematică a metodelor de cuantizare și a impactului per-layer.

**Experimente:**

1. **Per-tensor vs Per-channel INT8** — compară cele două granularități
2. **Sensitivity analysis** — cuantizează câte un layer pe rând, măsoară impactul individual
3. **Layer-wise error** — MSE/MAE între ponderile originale și cele cuantizate per-strat
4. **Timing** — latența de inferență FP32 vs FP16 vs INT8 pe MacBook
5. **Memory footprint** — dimensiunea modelului serializat în fiecare format

**Livrabile:** Script `layer_sensitivity_analysis.py`, plot-uri, tabel comparativ.

---

## Faza 4 — Scalare la modele mai mari + GPU NVIDIA

**Obiectiv:** Validarea concluziilor pe modele mai mari și pe hardware cu suport FP8 nativ.

**Pași:**

1. Repetă Fazele 1-3 pe `vit_small_patch16_224` (22M params) și `vit_base_patch16_224` (86M params)
2. Migrare pe Google Colab Pro / Kaggle cu GPU A100 sau H100
3. Compară: FP8 INT8 (software) vs `torch.float8_e4m3fn` (hardware nativ)
4. Măsoară speedup real pe Tensor Cores
5. (Opțional, dacă rămâne timp) Test pe un model generativ (FLUX.1 schnell)

**Livrabile:** Notebook Colab, rezultate pe GPU NVIDIA, comparație cross-hardware.

---

## Faza 5 — Redactare lucrare de licență

**Structura propusă:**

1. **Introducere** — Motivație, contribuții
2. **Fundamente teoretice** — ViT, formate numerice (FP32/FP16/FP8/INT8), tehnici de cuantizare
3. **Design experimental** — Metodologie, hardware, metrici
4. **Studiu preliminar** — CIFAR-10 (preluat din raportul existent)
5. **Cuantizare pe ImageNet** — Rezultatele principale (Fazele 1-3)
6. **Scalabilitate** — Modele mari + GPU NVIDIA (Faza 4)
7. **Concluzii** — Ghid practic de cuantizare, contribuții, limitări, direcții viitoare
8. **Bibliografie**

---

## Timeline estimativ

| Fază | Durată estimată | Dependențe |
|------|----------------|------------|
| Faza 1 — FP16 ImageNet | 1 săptămână | Model pretrained timm, subset ImageNet |
| Faza 2 — INT8 MatMul | 1-2 săptămâni | Faza 1 completă |
| Faza 3 — Comparații | 1-2 săptămâni | Faza 2 completă |
| Faza 4 — Scalare NVIDIA | 2-3 săptămâni | Fazele 1-3, acces Colab/Kaggle |
| Faza 5 — Redactare | 3-4 săptămâni | Toate fazele experimentale |

**Total estimat: 8–12 săptămâni**

---

## Recomandări practice

- **ImageNette** (10 clase din ImageNet, 9469 imagini) este un proxy excelent pentru teste rapide locale, înainte de validarea pe ImageNet complet
- Lucrează cu ChatGPT pentru caveat-urile specifice cuantizării ViT (conform recomandării coordonatorului)
- Păstrează tot codul CIFAR-10 funcțional — servește ca test de regresie
- Documentează fiecare experiment cu config YAML + JSON results (structura existentă e bună)
