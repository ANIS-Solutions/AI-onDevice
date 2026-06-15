# Edge AI Content Moderator

A lightweight, on-device content moderation system. It uses a fine-tuned Vision-Language Model (CLIP) optimized for mobile deployment via ONNX and FP16 quantization. The system applies a temporal Finite State Machine (FSM) for real-time video stabilization and utilizes Edge Data Condensation to send anonymized behavioral summaries to a cloud LLM.

##  Repository Structure

This repository is divided into specific environments for training, evaluation, edge deployment, and cloud processing.

### 1. `model_training/`
Contains the core pipeline to prepare the production weights.
* `train_lora.py`: Fine-tunes the CLIP Vision Encoder using Low-Rank Adaptation (LoRA) and a Multi-Label Sigmoid approach (BCEWithLogitsLoss).
* `export_onnx.py`: Splits the trained model into independent Text and Vision encoders, exporting them to ONNX (opset 14).
* `optimize_fp16.py`: Quantizes the exported Vision ONNX model from Float32 to Float16 to halve the memory footprint.
* `requirements.txt`: Dependencies to reproduce the training environment.

### 2. `evaluation/`
Contains scripts to validate the model's accuracy and the integrity of the quantization process.
* `generate_predictions.py`: Runs batch inference on test datasets (Images, Kinetics, UCF Crime) and saves raw probabilities to CSV.
* `calculate_metrics.py`: Reads the CSVs to compute Accuracy, Precision, Recall, F1-Score, and plots Confusion Matrices.
* `evaluate_video.py`: Simulates the Android edge pipeline, applying the FSM logic on local video files.
* `flipped_decisions_test.py`: Runs parallel inference between FP32 and FP16 models to verify exactly 0% quantization drift (no flipped safety decisions).

### 3. `cloud_backend/`
Handles the generation of behavioral reports.
* `generate_report.py`: Processes the 512-dimensional condensed embeddings received from the edge device. It calculates activity distributions via Cosine Similarity and invokes the Gemini 1.5 API to output a natural language behavioral report for parents.

### 4. `research_and_experiments/`
Archive of initial research, notebook iterations, and dataset preparation.
* Contains Jupyter Notebooks documenting the dataset harmonization process using Qwen2.5-VL and deprecated architectural tests (e.g., probability dilution in early Softmax experiments).

### 5. `android_app/` (Edge Client)
* The Android Studio project containing the Kotlin implementation of the ONNX Runtime, CameraX pipeline, FSM stabilization, and the K-Medoids/PCA clustering logic for Edge Condensation.

##  How to Run the ML Pipeline

To reproduce the model locally, navigate to `model_training/` and execute:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the LoRA adapters
python train_lora.py

# 3. Export graphs to ONNX
python export_onnx.py

# 4. Apply FP16 Quantization
python optimize_fp16.py