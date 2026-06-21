# research_and_experiments/

> Iterative research archive documenting the data engineering decisions, experimental directions explored, and the reasoning behind what was adopted vs. discarded. These notebooks form the foundation that the production pipeline in `model_training/` is built on.

← [Back to root](../README.md)

---

## Directory Structure

```
research_and_experiments/
├── 01_initial_data_structuring.ipynb        # Raw dataset taxonomy, label compression, CLIP baseline
├── 02_knowledge_distillation_experiment.ipynb  # CLIP → MobileNetV3 distillation (not adopted)
└── 03_lvlm_data_harmonization.ipynb         # Qwen2-VL automated safety auditor → golden dataset
```

---

## Why This Archive Exists

Production ML systems are built on a trail of decisions — what was tried, what failed, and why. This directory preserves that trail. Each notebook represents a distinct research question that shaped the final architecture. Reproducing the full pipeline requires understanding these choices, not just the final code.

---

## `01_initial_data_structuring.ipynb`

**Research question:** The raw dataset arrives from multiple Roboflow sources with flat, inconsistent class labels. How do we establish a clean, unified taxonomy before any model training begins?

### Dataset

~18,500 images collected from Roboflow and adversarial web sources. Raw labels were noisy, overlapping, and inconsistent across sources — the same visual concept appeared under different names depending on the collection origin.

### Approach

A keyword-based compression function maps all raw column labels into six semantic groups:

```python
groups = {
    'SFW_Characters':   ['Mickey Mouse', 'SpongeBob', 'Minion', 'Peter Griffin', ...],  # 18 categories
    'NSFW_Content':     ['AdultContents', 'Nudity', 'Sexual Position', 'bikini', ...],  # 9 categories
    'Violence_Danger':  ['Gun', 'pistol', 'Violent', 'Harassment', 'gross'],
    'Objects_Nature':   ['airplane', 'car', 'clock', 'tree', 'umbrella', 'toy'],
    'Animals':          ['elephant', 'panda', 'tiger', 'zebra'],
    'Humans':           ['boy', 'person', 'player']
}
```

Each row takes the **max** across all matched columns per group, producing a clean multi-hot binary matrix. An `Others` column captures images that match none of the groups.

**Dataset composition after compression:**

```
SFW_Characters=0, NSFW_Content=0, Violence=0, ...Others=1    7,769  (largest single bucket)
SFW_Characters=0, NSFW_Content=0, Violence=1                 5,948
SFW_Characters=0, NSFW_Content=1                             2,748
SFW_Characters=1                                             1,393
... (mixed multi-label combinations)
```

### CLIP Zero-Shot Baseline

The notebook includes an early zero-shot scan using the pre-fine-tuned base CLIP model against the test folder. This produced an initial Safe/Unsafe split (281 Unsafe / 263 Safe out of 544 scanned) that validated the taxonomy and established the performance floor that the LoRA fine-tuning pipeline in `model_training/` needed to surpass.

### What this produced

A structured understanding of the dataset's class distribution and the identification of two key problems that led directly to the next two notebooks:
1. **Label ambiguity** — the same image could reasonably belong to multiple groups (e.g. a cartoon character in a violent scene)
2. **Label noise** — cross-source inconsistency meant the binary labels couldn't be trusted for fine-tuning without re-annotation

---

## `02_knowledge_distillation_experiment.ipynb`

**Research question:** Can a much smaller student model (MobileNetV3-Large) learn the CLIP teacher's safety classification ability via knowledge distillation, enabling even cheaper on-device inference than the FP16 ONNX CLIP encoder?

### Architecture

```
Teacher: CLIP ViT-B/32 + LoRA adapters (clip_v1_sports_finished)
              ↓ frozen at inference, used only to generate soft labels
Student: MobileNetV3-Large (ImageNet pretrained)
              classifier[3] replaced: Linear(960 → 2)
```

### Distillation Loss

```
L_total = α · KLDiv(σ(student/T) ‖ σ(teacher/T)) · T²  +  (1 - α) · CrossEntropy(student, hard_labels)

Config:  temperature T = 4.0
         alpha α = 0.5   (equal weight to soft and hard loss)
```

The temperature parameter `T=4.0` softens the teacher's probability distribution, making it more informative for the student than hard one-hot labels alone. The `T²` scaling factor corrects for the magnitude change introduced by dividing logits by T.

### Training Setup

- **Teacher text embeddings** pre-computed once: two prompts — `"Safe content"` and `"Unsafe content"` — encoded and cached before training begins
- **Teacher logits** computed per batch with `torch.no_grad()` using `get_image_features()` → cosine similarity against pre-computed text features → scaled by 100.0
- **Student transforms**: augmented (RandomCrop, HorizontalFlip, Rotation±20°, ColorJitter) for training; clean resize for validation
- **Stratified 90/10 split** on `final_label` — ensures class balance in both train and val sets
- Best checkpoint saved when validation accuracy improves (actual model selection unlike the early `train_lora.py` version)

```
Epochs: 10 | Batch: 32 | LR: 1e-4 | Optimizer: AdamW
```

### Outcome: Not Adopted

The distillation experiment was ultimately **not used in production**. The core problem: collapsing CLIP's rich semantic embedding space into a binary `Safe / Unsafe` MobileNet head caused unacceptable loss of nuance.

Specifically:
- `Safe_Contextual_Body` (a person in athletic clothing mid-sport) and `Explicit_Adult` (lingerie) were assigned the same teacher logit distribution by the binary text prompts — the student had no way to distinguish them
- The binary framing also made threshold tuning meaningless — there was no per-category control
- A MobileNet that generalizes well to ImageNet does not automatically generalize to the long tail of screen content types (UI menus, cartoon characters, text overlays) that appear in child device monitoring

The production pipeline retains the full CLIP LoRA approach with `BCEWithLogitsLoss` and independent per-class sigmoid, which handles all of these cases correctly. This notebook remains as a documented dead end — knowledge distillation to a binary classifier is the wrong abstraction for a multi-label safety problem.

---

## `03_lvlm_data_harmonization.ipynb`

**Research question:** The 21,340-sample dataset has inconsistent labels across Roboflow sources. Rather than manual re-annotation, can a Large Vision-Language Model act as an automated safety auditor to produce a trustworthy golden standard dataset?

**Figure 5.1** — Architecture of the automated dataset engineering and annotation pipeline.

```
Raw Data Collection          HITL & Constraint          VLM Annotation          Validation &
(Roboflow + web sources)  →  Engineering             →  Engine               →  Final Output
 ~21K images                  Human-in-loop              Qwen2.5-VL (7b)         Parse JSON
 SFW characters               Strict constraint                                   Quality Filter
 Adversarial edge cases        based prompts                                      Multi-Hot
                                                                                  Conversion
                                                                                  → Golden Dataset
```

### Why LVLM Annotation?

Cross-source labels are structurally unreliable. A `Revealing_Clothing` label in one Roboflow dataset may correspond to athletic wear; in another, to lingerie. A model fine-tuned on these raw labels inherits the inconsistency. The solution is to discard all original labels and re-annotate the entire dataset from visual content alone using a model with strong vision-language reasoning.

### Auditor Model

**Qwen2-VL (7B)** — a Large Vision-Language Model deployed with 4-bit quantization (BitsAndBytesConfig) to fit within GPU memory constraints. The model receives each image alongside a strict constraint-based prompt engineered through Human-in-the-Loop (HITL) iteration.

### Output Schema

Each image is re-annotated with three fields:

```json
{
  "safety_category": "Revealing_Clothing",
  "safety_reason": "The individual is wearing a revealing outfit that exposes significant skin.",
  "clip_description": "A person in revealing clothing sits on a wooden surface."
}
```

The `clip_description` field is specifically designed for CLIP alignment — a neutral, factual visual description that the CLIP vision encoder's embedding space can match against text prompts without the ambiguity of safety-loaded language.

### Safety Categories Produced

| Category | Example |
|---|---|
| `Safe_Child_Friendly` | Cartoon characters, everyday objects, sports |
| `Revealing_Clothing` | Athletic wear, contextual body exposure |
| `Explicit_Adult` | Nudity, sexual content |
| `Violence_Scary` | Firearms, physical violence, blood |

### Scale

**21,340 images** re-annotated across all categories. Dataset excerpt from the output:

```
Row 0:    Revealing_Clothing  — "person wearing shorts with legs visible"
Row 1:    Explicit_Adult      — "woman with tattoo engaged in adult act"
Row 4:    Safe_Child_Friendly — "cartoon dog and cat interact with a woman"
Row 21335: Violence_Scary     — "hand holding a Sig Sauer P220 firearm"
```

### What This Produced

`final_dataset_v3_ready.csv` — the golden standard dataset used for all downstream training in `model_training/train_lora.py`. The data-centric re-annotation approach is the primary reason the final model achieves high diagnostic recall despite the dataset's multi-source origins. Fine-tuning on clean labels from a consistent annotation policy is more effective than fine-tuning on more data with noisy labels.

### Multi-Hot Vector Conversion

After LVLM annotation, the string categories are converted to binary multi-hot vectors compatible with `BCEWithLogitsLoss`:

```python
# In train_lora.py — MultiLabelCLIPDataset.__getitem__
if label in ["Safe_General", "Safe_Contextual_Body"]:   label_vector[0] = 1.0
elif label == "Unsafe_Violence":                         label_vector[2] = 1.0
elif label == "Unsafe_Sexual":                           label_vector[4] = 1.0
else:                                                    label_vector[1] = 1.0
```

---

## Research Timeline

```
Notebook 01            Notebook 03              Notebook 02          model_training/
Dataset taxonomy  →    LVLM re-annotation  →    Distillation    →    LoRA fine-tuning
                        (produces golden         (explored,           (production)
                         standard CSV)            not adopted)
```

The notebooks are numbered by research phase, not strictly by chronological execution. Notebook 03 (LVLM harmonization) must produce `final_dataset_v3_ready.csv` before `model_training/train_lora.py` can run. Notebook 02 (distillation) is an independent experiment that ran in parallel and informed the decision to retain the full CLIP architecture.
