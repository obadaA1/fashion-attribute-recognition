# Fashion Attribute Recognition

Multi-attribute classification of fashion images (material, color, pattern, texture) using CLIP-based pseudo-labelling and a multi-head deep learning classifier.

**Mini Project 2 — Deep Learning course, Spring 2026**

## 🎯 Project overview

This project trains a multi-branch classifier that predicts four visual attributes from a single fashion image:

| Attribute | Classes | Count |
|---|---|---|
| Material | cotton, silk/satin, leather, synthetic, denim, knitwear | 6 |
| Color | black, white, red, navy, beige, blue, green, yellow, grey, brown, pink, purple | 12 |
| Pattern | solid, striped, checkered/plaid, floral, geometric, animal print | 6 |
| Texture | smooth, rough/coarse, fluffy/furry, ribbed/structured, sheer | 5 |

## 🔬 Approach

1. **Data:** ~5,000 images sampled stratified from the TextileNet fabric dataset (23 folders)
2. **Pseudo-labelling:** Marqo-FashionSigLIP (fashion-domain-tuned SigLIP) with 5-view multi-view consensus (4/5 agreement required)
3. **Classifier:** ImageNet-pretrained backbone (ResNet-50 / EfficientNet-B3) with 4 classification heads, fine-tuned end-to-end
4. **Evaluation:** Per-attribute accuracy, per-class F1, confusion matrices on held-out test set
5. **Demo:** Streamlit app for single-image inference

## 📁 Repository structure

├── configs/          # YAML configs (paths, hyperparameters, prompts)
├── src/              # Python source (importable modules)
│   ├── data/         # sampling, pseudo-labelling, manifests, datasets
│   ├── models/       # multi-head classifier architectures
│   ├── training/     # training loop, losses
│   ├── evaluation/   # metrics, error analysis
│   ├── inference/    # prediction code for the demo
│   └── utils/        # seeds, logging helpers
├── scripts/          # CLI entrypoints (01_sample, 02_label, 03_train, 04_eval)
├── notebooks/        # EDA + Colab runners (thin wrappers around src/)
├── tests/            # pytest unit tests
└── docs/             # model card, dataset card, design decisions

## 🚀 Quick start

_Coming soon — will be populated as the pipeline is built._

## 📊 Experiment tracking

All training runs logged to [Weights & Biases](https://wandb.ai). Links in the model card once runs are completed.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

Obada Alsohli — [GitHub](https://github.com/obadaA1)
