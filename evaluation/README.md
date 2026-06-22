# evaluation/

> Academic benchmarking suite for the ANIS on-device content moderation system. Covers static image metrics, real-time FSM video evaluation, dynamic policy generation, and a rigorous quantization drift audit comparing FP32 and FP16 model decisions.

← [Back to root](../README.md)

---

## Directory Structure

```text
evaluation/
├── generate_baselines_onnx.py   # Compiles plain-text policies → JSON embeddings via Text Encoder
├── generate_predictions.py      # ONNX inference pipeline → dynamic probability CSVs
├── calculate_metrics.py         # Accuracy / Precision / Recall / F1 + confusion matrix plots
├── evaluate_video.py            # Real-time FSM video inference with dynamic HUD overlay
└── flipped_decisions_test.py    # FP32 vs FP16 parallel inference → quantization drift audit

```

---

## Prerequisites

All scripts consume outputs from `model_training/`. Run the training pipeline first and ensure the following files are available:

```text
model_training/
├── text_model_single.onnx         # FP32 text encoder (cloud / reference)
├── vision_model_single_fp32.onnx  # FP32 vision encoder (baseline for audit)
└── vision_model_fp16.onnx         # FP16 quantized vision encoder (edge deployment)

```

The inference scripts require `baselines.json` — the serialized Semantic JSON Config. This is generated locally using `generate_baselines_onnx.py` to simulate the server-side policy compilation process.

---

## Scripts

### `generate_baselines_onnx.py`

Simulates the Stage 1 Policy Engineering server. Generates baseline text embeddings for an infinite number of custom, dynamic policy categories using the FP32 ONNX text encoder.

**How it works:**

1. Loads a customizable dictionary of `POLICIES` (e.g., safe baselines, violence, adult, toxic text).
2. Runs the descriptions through `text_model_single.onnx` to generate L2-normalized 512D embeddings.
3. Serializes the matrices into `baselines.json`, which acts as the offline rulebook for all subsequent edge evaluations.

**Run:**

```bash
python evaluation/generate_baselines_onnx.py

```

**Output:** `baselines.json`

---

### `generate_predictions.py`

Runs the full `DynamicEdgeModerator` inference pipeline across static image test sets and writes raw per-class probability scores to CSV. Automatically adapts to any number of threat categories defined in the JSON.

**How it works:**

1. Loads text embeddings from `baselines.json`. Automatically routes any category containing `"SAFE"` in its key to the baseline matrix, and all others to the threat matrices.
2. For each test image: preprocesses using CLIP normalization constants, runs the FP16 ONNX vision encoder, and L2-normalizes the output embedding.
3. Computes **pairwise probability scaling** against the highest-scoring safe baseline:

```python
diff = clip((sim_threat - best_safe_sim) * 100.0, -50, 50)
prob = 1.0 / (1.0 + exp(-diff)) * 100.0

```

4. Saves ground truth labels and dynamic per-class probabilities to CSV.

**Run:**

```bash
python evaluation/generate_predictions.py

```

**Output:** `results_dynamic_images.csv`

```text
filename | ground_truth_label | THREAT_VIOLENCE_prob | THREAT_ADULT_prob | ...

```

---

### `calculate_metrics.py`

Reads `results_dynamic_images.csv` and evaluates classification performance across three probability thresholds. It dynamically detects any column ending in `_prob` and aggregates them (flagging an image as `BLOCKED` if *any* threat crosses the threshold).

**Run:**

```bash
python evaluation/calculate_metrics.py
# Requires: results_dynamic_images.csv

```

**Output:** `dynamic_thresholds_evaluation.png` — a three-panel confusion matrix grid

**Thresholds evaluated:** `55%`, `65%`, `85%`

Each threshold produces:

| Metric | Description |
| --- | --- |
| Accuracy | Overall correct classifications |
| Precision | Of all frames flagged as unsafe, how many truly were |
| Recall | Of all truly unsafe frames, how many were caught |
| F1-Score | Harmonic mean of precision and recall |

**Why three thresholds?** The 55% threshold maximizes recall (catches more unsafe content at the cost of false alarms). The 85% threshold maximizes precision (only flags high-confidence threats). The 65% threshold is the operational balance used in production.

---

### `evaluate_video.py`

Full real-time dynamic FSM pipeline on a local video file. Loads `baselines.json`, processes frames at a steady 2 FPS heartbeat, applies the Temporal FSM, and renders a dynamic HUD listing every active threat category onto the output video.

**Run:**

```bash
python evaluation/evaluate_video.py
# Input:  test_input.mp4
# Output: test_output_dynamic.mp4

```

**Pipeline per frame:**

```text
Video frame
    │
    ▼ (Steady Heartbeat: frame_idx % max(1, fps/2) == 0)
PIL Image → CLIP processor → FP16 pixel tensor
    │
    ▼
ONNX FP16 Vision Session → image embedding
    │
    ▼
Dynamic Pairwise cosine similarity vs. N threat matrices in JSON
    │
    ▼
FSM state transition
    │
    ▼
cv2.rectangle (Red Tint) if BLOCKED
+ Dynamic HUD overlay (state label + N per-class probabilities)
    │
    ▼
cv2.VideoWriter → output frame

```

**Finite State Machine (FSM):**

| Parameter | Default | Description |
| --- | --- | --- |
| `TARGET_FPS` | 2 | The steady edge-device heartbeat rate |
| `TRIGGER_TICKS` | 2 frames | Requires 1.0 second of consecutive threat before blocking |
| `RELEASE_TICKS` | 3 frames | Requires 1.5 seconds of consecutive safety before unblocking |

---

### `flipped_decisions_test.py`

Runs **parallel inference** on both the FP32 (`vision_model_single_fp32.onnx`) and FP16 (`vision_model_fp16.onnx`) vision models across the same test images. Uses the dynamic JSON policy to compute per-image probability drift and detects **flipped decisions** — cases where quantization caused a safety threshold to be crossed.

**Run:**

```bash
python evaluation/flipped_decisions_test.py
# Requires: test_images/ directory

```

**How it works:**

For each test image, the script:

1. Runs inference through the **FP32 session**.
2. Runs inference through the **FP16 session**.
3. Computes absolute probability drift per category: `|prob_fp32 - prob_fp16|`
4. Detects threshold crossings and classifies them:

| Flip type | Description | Severity |
| --- | --- | --- |
| `Safe → Threat (False Alarm)` | FP16 triggers where FP32 would not | Medium — UX disruption |
| `Threat → Safe (Missed)` | FP32 triggers but FP16 misses | **Critical** — safety failure |

**Expected result:** Zero flipped decisions. The production model achieves absolute stability, allowing the FP16 quantization step to safely halve the memory footprint to ~173 MB without compromising dynamic threshold decisions.

---

## Evaluation Pipeline — Run Order

```bash
# Step 1: Compile custom plain-text policy to JSON embeddings
python evaluation/generate_baselines_onnx.py
# → baselines.json

# Step 2: Generate dynamic predictions on image dataset
python evaluation/generate_predictions.py
# → results_dynamic_images.csv

# Step 3: Compute metrics and plot dynamic confusion matrices
python evaluation/calculate_metrics.py
# → dynamic_thresholds_evaluation.png

# Step 4: Validate quantization stability (FP32 vs FP16 drift audit)
python evaluation/flipped_decisions_test.py
# → console report

# Step 5: Visual dynamic FSM demo on video
python evaluation/evaluate_video.py
# → test_output_dynamic.mp4

```