

## Q1 — Vision Transformer on CIFAR-10 (PyTorch)

### Goal

Train a Vision Transformer (ViT) on the **CIFAR-10** dataset to achieve the best test accuracy.
Paper: *An Image is Worth 16×16 Words* (Dosovitskiy et al., ICLR 2021)

### How to Run in Colab

1. Open `q1.ipynb` in **Google Colab**.
2. Go to **Runtime → Change runtime type → GPU**.
3. Run all cells top → bottom.

Everything installs automatically and executes end-to-end.

### Model Setup

* Image patchify (16×16)
* Learnable positional embeddings
* `[CLS]` token prepended
* Transformer encoder blocks (MHSA + MLP + residual + norm)
* Classification head from `[CLS]` token

### Best Config

| Setting        | Value            |
| -------------- | ---------------- |
| Patch size     | 16×16            |
| Embedding dim  | 256              |
| Heads          | 8                |
| Encoder blocks | 6                |
| MLP hidden dim | 512              |
| Optimizer      | AdamW            |
| Scheduler      | Cosine annealing |
| Epochs         | 50               |
| Batch size     | 128              |

### Results

| Metric   | Test Accuracy (%) |
| -------- | ----------------- |
| CIFAR-10 | **83.39 %**        |

### Quick Notes / Analysis

* Smaller patches → more compute, no gain.
* 6 encoder blocks hit sweet spot for accuracy vs speed.
* Light augmentations (+1 % boost).
* AdamW + cosine schedule stabilized training.

---

## Q2 — Text-Driven Image Segmentation with SAM 2

### Goal

Perform **text-prompted segmentation** on a single image using **SAM 2**, optionally extendable to video.

### Pipeline

1. Load image
2. Take text prompt (e.g., “a dog”, “a car”)
3. Use **GroundingDINO/CLIPSeg** to find regions
4. Convert to region seeds
5. Pass to **SAM 2**
6. Display mask overlay

### How to Run in Colab

1. Open `q2.ipynb` in **Colab**.
2. Run the first install cells (auto installs SAM 2 + dependencies).
3. Upload an image or use the default.
4. Type a text prompt — output will show the final segmentation mask.

### Limitations

* Prompt sensitivity (“person” vs “man with hat”) affects results.
* Large or busy images may overlap masks.
* Free Colab GPU is slow for large models.

### Bonus (Video Extension)

Optional demo: text-driven video object segmentation (10–30 s clip) using mask propagation with SAM 2.

---

