# evaluation/

> Academic benchmarking suite for the ANIS on-device content moderation system. Covers static image metrics, real-time FSM video evaluation, and a rigorous quantization drift audit comparing FP32 and FP16 model decisions.

← [Back to root](../README.md)

---

## Directory Structure

```
evaluation/
├── generate_predictions.py      # ONNX inference pipeline → raw probability CSVs
├── calculate_metrics.py         # Accuracy / Precision / Recall / F1 + confusion matrix plots
├── evaluate_video.py            # Real-time FSM video inference with blur overlay rendering
└── flipped_decisions_test.py    # FP32 vs FP16 parallel inference → quantization drift audit
```

---

## Prerequisites

All scripts consume outputs from `model_training/`. Run the training pipeline first and ensure the following files are available:

```
model_training/
├── text_model.onnx              # FP32 text encoder (cloud / reference)
├── vision_model.onnx            # FP32 vision encoder (baseline)
└── vision_model_fp16.onnx       # FP16 quantized vision encoder (edge deployment)
```

The inference scripts also require `saved_embeddings.json` — the serialized Semantic JSON Config produced by the policy engineering server (see [AI-hosted](https://github.com/ANIS-Solutions/AI-hosted)).

---

## Scripts

### `generate_predictions.py`

Runs the full `EdgeModerator` inference pipeline across static image test sets and writes raw per-class probability scores to CSV.

**How it works:**

1. Loads text embeddings from `saved_embeddings.json` (the on-device policy config)
2. For each test image: preprocesses using CLIP normalization constants (`mean=[0.48145, 0.45782, 0.40821]`, `std=[0.26862, 0.26130, 0.27577]`), runs the FP16 ONNX vision encoder, L2-normalizes the output embedding
3. Computes **pairwise probability scaling** against each threat category:

```python
diff = clip((sim_threat - sim_safe) * LOGIT_SCALE, -50, 50)
prob = 1.0 / (1.0 + exp(-diff))
```

4. Classifies as `BLOCKED` if any threat probability exceeds the default threshold (85%)
5. Saves ground truth labels, predicted status, and per-class probabilities to CSV

**Run:**

```bash
python evaluation/generate_predictions.py
```

**Output:** `results_images.csv`

```
filename | ground_truth_label | predicted_status | Adult_prob | Violence_prob
```

**Notes:**
- Supports CUDA via `CUDAExecutionProvider` with CPU fallback
- Ground truth is read from `train_split.csv` alongside the test image folder
- Images with `ground_truth_label = Unknown` are excluded from metric calculation

---

### `calculate_metrics.py`

Reads `results_images.csv` and evaluates classification performance across three probability thresholds. Generates confusion matrix visualizations for each threshold.

**Run:**

```bash
python evaluation/calculate_metrics.py
# Requires: results_images.csv (from generate_predictions.py)
```

**Output:** `images_thresholds_evaluation.png` — a three-panel confusion matrix grid

**Thresholds evaluated:** `55%`, `65%`, `85%`

Each threshold produces:

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Precision | Of all frames flagged as unsafe, how many truly were |
| Recall | Of all truly unsafe frames, how many were caught |
| F1-Score | Harmonic mean of precision and recall |

**Label mapping:**

```python
'Unsafe_*'  → 1  (positive / unsafe)
'Safe_*'    → 0  (negative / safe)
```

**Why three thresholds?** The 55% threshold maximizes recall (catches more unsafe content at the cost of false alarms). The 85% threshold maximizes precision (only flags high-confidence threats). The 65% threshold is the operational balance used in production. All three are evaluated so the threshold can be tuned per strictness policy level.

---

### `evaluate_video.py`

Full real-time FSM pipeline on a local video file. Pre-computes text embeddings, processes frames at ~3 FPS, applies the four-state Temporal FSM, and renders the blur overlay and probability HUD onto the output video.

**Run:**

```bash
python evaluation/evaluate_video.py
# Input:  test_input.mp4
# Output: test_output.mp4
```

**Pipeline per frame:**

```
Video frame
    │
    ▼ (every ~3 FPS: frame_idx % max(1, fps/3) == 0)
PIL Image → CLIP processor → FP16 pixel tensor
    │
    ▼
ONNX FP16 Vision Session → image embedding (FP32 cast)
    │
    ▼
Pairwise cosine similarity vs. pre-computed text embeddings
    │
    ▼
Per-category threat probability (log-space stable sigmoid)
    │
    ▼
FSM state transition
    │
    ▼
cv2.GaussianBlur (kernel 99×99) if BLURRED or PENDING_RELEASE
+ HUD overlay (state label + per-class probabilities)
    │
    ▼
cv2.VideoWriter → output frame
```

**Finite State Machine:**

```
         ┌── N consecutive unsafe frames ──▶ PENDING_BLUR ──▶ BLURRED
SAFE ────┤                                                        │
         └◀── M consecutive safe frames ── PENDING_RELEASE ◀─────┘
```

| Parameter | Default | Description |
|---|---|---|
| `TRIGGER_DELAY_MS` | 400 ms | Time of consecutive unsafe frames before blur activates |
| `RELEASE_DELAY_MS` | 3000 ms | Time of consecutive safe frames before blur clears |
| Processing rate | ~3 FPS | `frame_idx % max(1, fps/3)` |

**Threat categories and thresholds:**

| Category | Threshold | Prompt excerpt |
|---|---|---|
| Adult | 0.40 | `"adult themes, sexual revealing clothing, lingerie..."` |
| Violence | 0.45 | `"Graphic physical violence, aggressive combat, blood, weapons..."` |

A safe baseline prompt is also encoded: `"Safe, innocent, everyday objects, normal content, people smiling."` — the pairwise scaling compares each threat logit directly against this baseline rather than using a global softmax.

**Note on text embedding pre-computation:** Text embeddings are computed once at startup using the FP32 ONNX text session and cached in memory for the duration of the video. This matches the production deployment model where embeddings are generated server-side and shipped in the policy JSON.

---

### `flipped_decisions_test.py`

Runs **parallel inference** on both the FP32 and FP16 vision models across the same 100 test images. Computes per-image probability drift and detects **flipped decisions** — cases where quantization caused a safety threshold to be crossed in either direction.

**Run:**

```bash
python evaluation/flipped_decisions_test.py
# Requires: test_images_sample/ directory with ≥100 PNG/JPG images
```

**How it works:**

For each test image, the script:

1. Runs inference through the **FP32 session** (`vision_model.onnx`)
2. Runs inference through the **FP16 session** (`vision_model_fp16.onnx`)
3. The session wrapper auto-detects the expected input dtype from the ONNX graph:

```python
expected_type = session_dict['vision'].get_inputs()[0].type
target_np_type = np.float16 if "float16" in expected_type else np.float32
```

4. Computes absolute probability drift per category: `|prob_fp32 - prob_fp16|`
5. Detects threshold crossings and classifies them:

| Flip type | Description | Severity |
|---|---|---|
| `Safe → Threat (False Alarm)` | FP16 triggers where FP32 would not | Medium — UX disruption |
| `Threat → Safe (Missed)` | FP32 triggers but FP16 misses | **Critical** — safety failure |

**Output report:**

```
==================================================
Final Quantization Stability Report:
Average Probability Drift: X.XXXX%
Total Flipped Decisions: N out of 200 checks
==================================================
```

For any flipped cases detected, the script prints the image filename, category, flip type, and original vs. quantized probability side-by-side.

**Expected result:** Zero flipped decisions. The production model achieved 0% quantization drift on safety-critical decisions — the FP16 quantization step halves the memory footprint from ~346 MB to 173 MB with no measurable safety regression.

---

## Evaluation Pipeline — Run Order

```bash
# Step 1: Generate raw predictions
python evaluation/generate_predictions.py
# → results_images.csv

# Step 2: Compute metrics and plot confusion matrices
python evaluation/calculate_metrics.py
# → images_thresholds_evaluation.png

# Step 3: Validate quantization stability (independent — no CSV dependency)
python evaluation/flipped_decisions_test.py
# → console report

# Step 4: Visual FSM demo on video (independent)
python evaluation/evaluate_video.py
# → test_output.mp4
```

---

## Interpreting Results

**Which threshold to use in production?** Match it to the policy strictness level set by the parent in Stage 1. The `evaluate_video.py` script uses per-category thresholds (Adult: 0.40, Violence: 0.45) which correspond to a Moderate policy. Strict policies lower these values; Relaxed policies raise them.

**What counts as a safety-critical flip?** Only `Threat → Safe (Missed)` flips in `flipped_decisions_test.py`. A `Safe → Threat (False Alarm)` is a UX problem (unnecessary blur), but a missed detection is a content safety failure. The audit specifically flags both but the zero-flip target applies primarily to the missed-detection direction.
