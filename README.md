# Multi-Head Fashion Attribute Recognition

Self-supervised backbones (DINOv2) + VLM-audited silver labels for joint per-image classification of **color, pattern, material, and texture** on a 13,355-image TextileNet subset.

**Mean test macro-F1: 0.765** (DINOv2 ViT-B/14) vs **0.649** (ResNet-50, identical recipe) — **+11.6 pp** absolute. DINOv2 wins on every head and on all 29 classes.

📄 [IEEE writeup (PDF)](./report.pdf) · 📓 [Notebook](./fashion_attributes.ipynb) · 🔬 [Provenance manifest](./manifest.json)

---

## TL;DR

Public fashion datasets each cover at most two of {color, pattern, material, texture}; none label all four jointly per image. I built a 13,355-image TextileNet subset and hybrid-labeled it: rule-based mapping for material (perfect-precision by construction), Marqo-FashionSigLIP zero-shot with 5-template prompt ensembling and confidence-threshold abstain for color / pattern / texture. A 210-image stratified human audit anchors silver-label trust at 0.971 / 0.803 / 0.681 macro-F1. A DINOv2 ViT-B/14 backbone with four parallel linear heads reaches mean test macro-F1 0.765 — uniformly +11.6 pp over an identically-recipe-trained ResNet-50. Cramér's V on material×texture (V_gt=0.318, V_pred=0.353, Δ=+0.035) confirms the multi-head design preserves rather than collapses the joint distribution.

## Method

- **Backbone:** DINOv2 ViT-B/14 (86.6M params, self-supervised on LVD-142M)
- **Heads:** 4 parallel linear (768 → {6, 12, 6, 5}; 22.3k params)
- **Loss:** sum of 4 class-weighted CE, abstained labels masked via `ignore_index=-100`
- **Schedule:** 20 epochs, AdamW, discriminative LRs (1e-5 backbone / 1e-4 heads), 2-epoch warmup + 18-epoch cosine, AMP fp16, batch 64
- **Hardware:** single NVIDIA T4

## Results

| Head     | n eval | Acc.  | DINOv2 F1 | ResNet-50 F1 | Δ      |
|----------|-------:|------:|----------:|-------------:|-------:|
| material |  3,760 | 0.747 |     0.744 |        0.612 | +13.2  |
| color    |  3,743 | 0.842 |     0.827 |        0.738 |  +8.9  |
| pattern  |  3,527 | 0.822 |     0.737 |        0.616 | +12.1  |
| texture  |  3,487 | 0.742 |     0.753 |        0.630 | +12.3  |
| **mean** |    —   | **0.788** | **0.765** | **0.649** | **+11.6** |

Same data, splits, augmentation, optimizer, schedule, class weights. Only the backbone changes.

## Reproduction
git clone https://github.com/obadaA1/fashion-attribute-recognition
cd fashion-attribute-recognition
pip install -r requirements.txt