# Model Training & Edge Optimization Pipeline

This directory contains the core machine learning pipeline for fine-tuning the Vision-Language Model (VLM) and optimizing it for Edge deployment (Android). 

The pipeline transitions a base CLIP model into a highly specialized, asymmetric Multi-Label classifier using Low-Rank Adaptation (LoRA), and prepares it for real-time mobile inference via ONNX and FP16 quantization.

##  Directory Structure

* `train_lora.py`: The primary fine-tuning script. It trains the CLIP Vision Encoder using LoRA and applies a Multi-Label Sigmoid approach (BCEWithLogitsLoss) to prevent the probability dilution inherent in standard Softmax architectures.
* `export_onnx.py`: Extracts the fine-tuned weights and explicitly splits the architecture into independent Text and Vision encoders. Exports the computational graphs to ONNX format (opset 14) for maximum compatibility with Android Neural Networks API (NNAPI).
* `optimize_fp16.py`: Performs post-training quantization. It converts the exported Vision ONNX model from Float32 to Float16, effectively halving the memory footprint while maintaining semantic accuracy.
* `requirements.txt`: Contains the exact versions of the libraries required to reproduce the training environment.

##  How to Reproduce

To train the model from scratch and generate the Edge-ready ONNX files, follow these steps in order:

### 1. Environment Setup
Ensure you are using Python 3.9+ and install the required dependencies:
```bash
pip install -r requirements.txt

```

### 2. Data Preparation

Before starting the training process, ensure your dataset is correctly placed in the root of this directory:

* `final_dataset_v3_ready.csv`: The harmonized labeling CSV.
* `train/`: The directory containing the actual raw images.
*(Note: Datasets are excluded from this repository via `.gitignore` due to size and privacy constraints).*

### 3. Execution Pipeline

**Step A: Fine-Tune the Model**
Run the training script to inject LoRA adapters and learn the child-safety semantic boundaries.

```bash
python train_lora.py

```

*(This will generate the `clip_v2_sigmoid_best` directory containing the PEFT weights).*

**Step B: Export to ONNX**
Split the trained model and export the static computational graphs.

```bash
python export_onnx.py

```

*(This will generate `text_model.onnx` and `vision_model.onnx`).*

**Step C: Edge Optimization (FP16)**
Compress the Vision model for mobile inference.

```bash
python optimize_fp16.py

```

*(This will generate `vision_model_fp16.onnx`).*

## Expected Outputs for Android Client

After completing the pipeline, the following two files must be copied to the Android application's `assets/` folder:

1. `text_model.onnx` (Static Float32 embeddings generator)
2. `vision_model_fp16.onnx` (Dynamic Float16 real-time inference engine)
