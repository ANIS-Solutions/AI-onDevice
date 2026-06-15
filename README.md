![](./_asset/anisChildApp-asset.jpg)

# AI-Moderator-Pipeline

A research-driven machine learning pipeline for on-device child safety moderation and semantic activity analysis. This system leverages Multi-Label Sigmoid architectures, LoRA fine-tuning, and Large Vision-Language Models for dataset harmonization to ensure high-fidelity threat detection with zero-shot generalization.


## Repository Structure

* **`model_training/`**: Pipeline for LoRA fine-tuning (BCEWithLogitsLoss), VLM architecture splitting, and FP16 ONNX optimization.
* **`evaluation/`**: Academic benchmarking suite including precision/recall metrics, confusion matrix plotting, and quantization drift analysis.
* **`research_and_experiments/`**: Iterative research archive including LVLM data harmonization (Qwen-VL) and distillation experiments.

---
## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-FF9900?logo=huggingface&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-005CED?logo=LoRA&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white)
![Qwen-VL](https://img.shields.io/badge/Qwen--VL-FF4500?logo=qwen&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-005CED?logo=openai&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Google Gemini API](https://img.shields.io/badge/Gemini-8E75B2?logo=google-gemini&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3776AB?logo=python&logoColor=white)

---

## Getting Started

1. Clone the repository:

```bash
git clone [https://github.com/ANIS-Solutions/AI-Moderator-Pipeline.git](https://github.com/ANIS-Solutions/AI-Moderator-Pipeline.git)
cd AI-Moderator-Pipeline

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

3. Configure environment variables (Gemini API Key):

```bash
export GEMINI_API_KEY="your_api_key_here"

```

4. Run the training pipeline:

```bash
cd model_training
python train_lora.py

```

---

## Development

```bash
# Run model training (LoRA Fine-tuning)
python model_training/train_lora.py

# Execute evaluation benchmarks (Recall/Precision Analysis)
python evaluation/calculate_metrics.py

# Run report generation backend (Gemini Integration)
python cloud_backend/generate_report.py

# Clean research workspace
rm -rf research_and_experiments/.ipynb_checkpoints

```

---

## Bug Reports

If you encounter performance regressions or pipeline failures, please open an Issue with the training logs or evaluation CSV reports attached.

*Bye! ... Bye!*

```
