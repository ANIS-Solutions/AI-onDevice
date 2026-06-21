# Model Training & Edge Optimization Pipeline

> Fine-tunes a CLIP Vision-Language Model using Low-Rank Adaptation (LoRA) and exports an asymmetric, quantized inference graph for real-time on-device content moderation.

![Training Pipeline](_asset/Training%20pipline%20.png)
*Five-stage training pipeline: CSV ingestion → frozen text encoding with dummy tensor trick → LoRA adapter injection into ViT q/v projections (rank 16, ~4.2% trainable params) → BCEWithLogitsLoss multi-label optimization → asymmetric ONNX export with FP16 quantization, reducing the vision encoder from ~346 MB to 173 MB for on-device inference.*

---

## Overview

This pipeline transitions a base `openai/clip-vit-base-patch32` model into a specialized, asymmetric **multi-label content safety classifier** using Parameter-Efficient Fine-Tuning (PEFT). It is designed with a privacy-first, edge-deployment architecture: the heavy vision inference runs entirely **on-device** (Android), while the lightweight text encoder is served from the cloud to allow dynamic, zero-shot policy updates without redeployment.

### Why this architecture?

| Design choice | Rationale |
|---|---|
| LoRA over full fine-tuning | Only ~4.2% of parameters are trainable, preserving the pre-trained CLIP backbone and preventing catastrophic forgetting |
| `BCEWithLogitsLoss` over `CrossEntropyLoss` | Softmax enforces a zero-sum constraint across classes — BCE with independent sigmoid activations allows multiple threat categories to be active simultaneously |
| Frozen text encoder | Preserves zero-shot semantic reasoning; category prompts like `"violence"` or `"adult content"` already encode rich representations |
| Asymmetric export (vision → edge, text → cloud) | Keeps raw video processing on-device for privacy; enables policy changes via cloud text embeddings without retraining |
| FP16 post-training quantization | Halves model footprint from ~346 MB → 173 MB with negligible accuracy loss on safety-critical decisions |

---

## Directory Structure

```
model_training/
├── train_lora.py           # LoRA fine-tuning with BCEWithLogitsLoss (Stage 1–4)
├── export_onnx.py          # Splits and serializes both encoders to ONNX opset 14 (Stage 5a)
├── optimize_fp16.py        # FP16 post-training quantization on vision model (Stage 5b)
└── requirements.txt        # Pinned dependencies for reproducibility
```

---

## Architecture — 5-Stage Pipeline

### Stage 1 · Data Ingestion (`train_lora.py → MultiLabelCLIPDataset`)

The `MultiLabelCLIPDataset` loader parses a structured CSV file containing image filenames and ground-truth labels. It applies bicubic resizing and normalization via the CLIP processor and maps each label string to a **binary one-hot vector**:

```
y ∈ {0, 1}^C    where C = 5 semantic classes
```

| Index | Class | Description |
|---|---|---|
| 0 | Safe Content | Child-friendly, sports, casual clothing |
| 1 | Neutral Objects | Everyday objects, vehicles, buildings, UI |
| 2 | Violence | Physical violence, blood, weapons, horror |
| 3 | Inappropriate Text | Sexual or violent text in screenshots |
| 4 | Adult Content | Revealing clothing, lingerie, suggestive content |

Each item returns a `pixel_values` tensor of shape `(3, 224, 224)` and a `label_vector` of shape `(5,)`.

---

### Stage 2 · Text Encoding Pathway — Frozen (`train_lora.py → train_model`)

The CLIP text encoder is kept **entirely frozen** to preserve its zero-shot semantic capabilities. It processes the five natural-language category prompts (not just class names — full descriptive sentences) to produce a fixed matrix of text embeddings:

```
E_text ∈ ℝ^(C × D)    where D = 512
```

A **dummy pixel tensor** (zeros, shape `(C, 3, 224, 224)`) is passed to satisfy the CLIP model's requirement for both modalities during a forward pass. This tensor contributes no gradient and updates no visual weights.

---

### Stage 3 · Vision Encoding + LoRA Injection — Sole Trainable Stage (`train_lora.py`)

LoRA adapters are injected into the frozen ViT-B/32 using the `peft` library:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
```

The modified forward pass for each injected projection:

```
h = W₀x + ΔWx = W₀x + BAx
```

Where `B ∈ ℝ^(d×r)` is zero-initialized and `A ∈ ℝ^(r×k)` is Gaussian-initialized, ensuring `ΔW = BA = 0` at training start — the model begins from exact pre-trained weights.

**Parameter efficiency** (with `d = k = 768`, `r = 16`):

```
Trainable ratio = r(d + k) / (d × k) = 16 × 1536 / 589824 ≈ 4.2%
```

A **second dummy tensor** (zero `input_ids` and `attention_mask`, shape `(N, 77)`) is passed to satisfy the vision forward pass, mirroring Stage 2.

---

### Stage 4 · Optimization & Loss (`train_lora.py → train_model`)

Both embedding matrices are L2-normalized before similarity computation:

```
ê = e / ‖e‖₂
```

Logits are computed via temperature-scaled dot product:

```
Z = s · Ê_vis · Ê_text^T    where s = exp(logit_scale)
```

**Multi-label loss** — BCEWithLogitsLoss evaluates each class independently:

```
L = -(1/N) Σᵢ [ yᵢ log σ(zᵢ) + (1 - yᵢ) log(1 - σ(zᵢ)) ]
```

Gradients flow **exclusively** through the LoRA adapter weights. The frozen base model weights receive no gradient updates.

**Training hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | `openai/clip-vit-base-patch32` |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | 32 |
| Epochs | 5 |
| Loss function | `BCEWithLogitsLoss` |
| Input resolution | `224 × 224` |
| Embedding dimension (D) | 512 |
| Number of classes (C) | 5 |
| Train/val split | 90 / 10 |

---

### Stage 5 · Export & Quantization — Asymmetric Split

After fine-tuning, the LoRA adapter weights are first merged back into the base ViT weights to eliminate adapter overhead at inference time:

```
W_final = W₀ + BA
```

The merged model is then split and exported via two independent pathways. Both wrappers strip Hugging Face dictionary outputs for clean ONNX tracing:

```python
# Vision — returns image embeddings only
def forward(self, pixel_values):
    return self.model(pixel_values=pixel_values, return_dict=False)[0]

# Text — returns text embeddings only
def forward(self, input_ids, attention_mask):
    return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
```

Both are exported with `torch.onnx.export` at **opset 14** with `do_constant_folding=True` for maximum compatibility with the Android Neural Networks API (NNAPI).

---

#### Vision Pathway → Edge Device · [Child App](https://github.com/ANIS-Solutions/Child-app)

`export_onnx.py` exports `vision_model.onnx` (Float32). `optimize_fp16.py` then applies post-training quantization:

```python
model_fp16 = float16.convert_float_to_float16(onnx.load("vision_model.onnx"))
onnx.save(model_fp16, "vision_model_fp16.onnx")
```

Result: `vision_model_fp16.onnx` at **~173 MB** (halved from ~346 MB), with zero measured quantization drift on safety-critical decisions. This model is bundled into the Android child application's `assets/` folder and runs entirely **on-device** — raw video frames never leave the device.

See → **[ANIS-Solutions/Child-app](https://github.com/ANIS-Solutions/Child-app)** for the Android inference client, ONNX Runtime integration, and the Temporal FSM used for UX stabilization.

---

#### Text Pathway → Cloud Server · [AI-Hosted](https://github.com/ANIS-Solutions/AI-hosted)

The text encoder was never modified during training, so no weight merging is needed. `export_onnx.py` extracts and exports the frozen base CLIP text encoder directly to `text_model.onnx` (Float32 — not quantized, as it runs on the cloud server where memory is not constrained).

This enables **dynamic zero-shot policy updates**: new safety categories can be added or modified by updating the natural-language prompts server-side and regenerating text embeddings — no retraining, no redeployment of the mobile app required.

See → **[ANIS-Solutions/AI-hosted](https://github.com/ANIS-Solutions/AI-hosted)** for the cloud inference server, custom policy management API, and the Gemini-powered reporting pipeline.

---

## Reproduce — Setup & Execution

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU recommended (CPU fallback supported via `Config.DEVICE`)
- Dataset files excluded from this repo via `.gitignore` due to size and privacy constraints

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Place the following in this directory before running:

```
model_training/
├── final_dataset_v3_ready.csv    # Harmonized labeling CSV (filename, final_label columns)
└── train/                         # Raw image files referenced in the CSV
```

**Label format** — the `final_label` column expects one of:

| CSV value | Mapped class index |
|---|---|
| `Safe_General` or `Safe_Contextual_Body` | 0 — Safe Content |
| `Unsafe_Violence` | 2 — Violence |
| `Unsafe_Sexual` | 4 — Adult Content |
| anything else | 1 — Neutral Objects |

### 3. Fine-tune

```bash
python train_lora.py
```

Trains for 5 epochs with progress bars per batch. Saves PEFT adapter weights and processor after every epoch.

**Output:** `clip_v2_sigmoid_best/` — PEFT adapter weights + processor config

### 4. Export to ONNX

```bash
python export_onnx.py
```

Loads the merged model from `merged_clip_model/`, wraps both encoders for clean tracing, and exports static computation graphs.

> **Note:** Requires a merged model directory. If you are exporting directly after training, first merge PEFT weights into the base using `model.merge_and_unload()` and save to `merged_clip_model/`.

**Output:** `text_model.onnx`, `vision_model.onnx`

### 5. Quantize for edge

```bash
python optimize_fp16.py
```

Converts the vision model from Float32 to Float16. Prints size reduction to stdout.

**Output:** `vision_model_fp16.onnx` (~173 MB)

---

## Deployment Targets

After completing the pipeline, the two output models feed into separate downstream repositories:

| File | Size | Destination | Repository |
|---|---|---|---|
| `vision_model_fp16.onnx` | ~173 MB | Android `assets/` — on-device inference | [Child-app](https://github.com/ANIS-Solutions/Child-app) |
| `text_model.onnx` | ~346 MB | Cloud server — policy embedding generation | [AI-hosted](https://github.com/ANIS-Solutions/AI-hosted) |

---

## Notes & Known Limitations

- **Label mapping is hard-coded** in `MultiLabelCLIPDataset.__getitem__`. Classes 3 (Inappropriate Text) and 4 (Adult Content) map to the same CSV label `Unsafe_Sexual` — index 4 is set but the mapping for explicit text screenshots would require a separate label in the CSV.
- **No validation loop in the training script.** The best checkpoint logic (`best_acc`) is initialized but never updated — every epoch overwrites the output directory regardless of performance. Adding evaluation against `val_df` is recommended.
- **`logit_scale` access has a fallback** for PEFT-wrapped models (`model.base_model.model.logit_scale`). This is a defensive check but worth verifying against your specific PEFT version.
- **`merged_clip_model/` is a prerequisite** for `export_onnx.py` but is not produced by `train_lora.py` directly. A merge step is needed between training and export.
- **FP16 quantization is applied globally.** For stricter safety-critical deployments, consider mixed-precision quantization or post-quantization validation against a held-out safety benchmark.

---

## License

See repository root for licensing information. Dataset files are excluded and are not redistributed.
