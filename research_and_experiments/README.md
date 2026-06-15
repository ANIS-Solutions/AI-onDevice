# Research & Experiments Archive

This directory serves as the historical archive of the scientific method applied during the development of the Edge AI Moderator. It rigorously documents the iterative research process, the sequence of our experiments, failed hypotheses, and the final dataset harmonization steps that informed the production architecture.

**Note to Reviewers:** The code in this directory is for archival and research demonstration purposes. For the finalized, production-ready training and edge-deployment pipelines, please refer to the `../model_training/` directory.

## Chronological Research Journey

### Phase 1: Initial Data Structuring
* **Notebook:** `01_initial_data_structuring.ipynb`
* **Focus:** Aggregating raw datasets and establishing the preliminary multi-label mappings (e.g., SFW_Characters, NSFW_Content, Violence_Danger). This phase highlighted the massive noise and mislabeling present in raw scraped datasets, proving that heuristic grouping was insufficient.

### Phase 2: The Knowledge Distillation Failure
* **Notebook:** `02_knowledge_distillation_experiment.ipynb`
* **Hypothesis:** We can compress the heavy Vision-Language Model (CLIP) for mobile edge deployment using a Knowledge Distillation paradigm (Teacher: CLIP, Student: MobileNetV3).
* **Outcome (Failed):** The core expectation was flawed. While the student model (MobileNet) successfully learned to mimic the teacher on the training set, it catastrophically failed to maintain the "zero-shot semantic flexibility" required to detect novel out-of-distribution threats. 
* **Pivot:** This critical failure forced the abandonment of the distillation approach. We pivoted toward retaining the robust CLIP architecture via LoRA fine-tuning and solving the mobile memory constraint through ONNX FP16 Post-Training Quantization instead.

### Phase 3: Advanced Data Harmonization via LVLM
* **Notebook:** `03_lvlm_data_harmonization.ipynb`
* **Focus:** Returning to the data layer after the distillation failure. We realized that highly accurate, context-aware training data was mandatory for the new V2 Pairwise Architecture.
* **Outcome:** Successfully utilized an advanced Large Vision-Language Model (`Qwen2-VL-7B-Instruct`) to automatically audit and re-classify thousands of ambiguous frames (e.g., distinguishing between explicit nudity and safe contextual body exposure in sports). This generated the highly curated dataset required for the final production model.