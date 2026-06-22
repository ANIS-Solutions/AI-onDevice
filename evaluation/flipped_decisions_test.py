"""
Module: flipped_decisions_test.py
Description: 
    Evaluates FP16 post-training quantization drift against the FP32 baseline.
    Uses a dynamic JSON policy to test if precision loss causes any False Alarms 
    or Missed Threats across user-defined categories.
"""

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
from tqdm import tqdm

class DriftConfig:
    TEST_IMAGES_DIR = "test_images"  # Directory containing test images
    NUM_IMAGES_TO_TEST = 1000 
    
    # Vision models to compare
    FP32_VISION = "vision_model_single_fp32.onnx"
    FP16_VISION = "vision_model_fp16.onnx"
    
    POLICY_JSON = "baselines_2.json" # Source of truth for dynamic categories
    THRESHOLD = 65.0
    LOGIT_SCALE = 100.0

def preprocess(image_path, target_np_type):
    frame = cv2.imread(image_path)
    if frame is None: return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(target_np_type)

def get_inference_probs(session, tensor, safe_matrices, threat_matrices):
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: tensor})[0]
    img_embed = out[0].astype(np.float32)
    img_embed = img_embed / np.linalg.norm(img_embed)

    # Max similarity across all safe categories
    safe_sims = [np.max(np.dot(img_embed, mat)) for mat in safe_matrices if mat.shape[0] > 0]
    best_safe_sim = max(safe_sims) if safe_sims else -1.0

    probs = {}
    for name, mat in threat_matrices.items():
        sim_threat = np.max(np.dot(img_embed, mat))
        diff = np.clip((sim_threat - best_safe_sim) * DriftConfig.LOGIT_SCALE, -50, 50)
        probs[name] = (1.0 / (1.0 + np.exp(-diff))) * 100.0
        
    return probs

def evaluate_quantization_drift():
    if not os.path.exists(DriftConfig.TEST_IMAGES_DIR):
        print(f"Error: Directory '{DriftConfig.TEST_IMAGES_DIR}' not found.")
        return

    with open(DriftConfig.POLICY_JSON, 'r', encoding='utf-8') as f:
        policy = json.load(f)

    # Separate safe and threat matrices dynamically
    safe_matrices = [np.array(list(data.values()), dtype=np.float32).T for k, data in policy.items() if "SAFE" in k.upper()]
    threat_matrices = {k: np.array(list(data.values()), dtype=np.float32).T for k, data in policy.items() if "SAFE" not in k.upper()}

    fp32_sess = ort.InferenceSession(DriftConfig.FP32_VISION, providers=['CPUExecutionProvider'])
    fp16_sess = ort.InferenceSession(DriftConfig.FP16_VISION, providers=['CPUExecutionProvider'])

    image_files = [f for f in os.listdir(DriftConfig.TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:DriftConfig.NUM_IMAGES_TO_TEST]
    
    flipped_cases = []
    drifts = []

    print(f"Analyzing {len(image_files)} images for Quantization Drift across {len(threat_matrices)} dynamic categories...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(DriftConfig.TEST_IMAGES_DIR, img_name)
        
        tensor_fp32 = preprocess(img_path, np.float32)
        tensor_fp16 = preprocess(img_path, np.float16)
        if tensor_fp32 is None: continue

        probs_orig = get_inference_probs(fp32_sess, tensor_fp32, safe_matrices, threat_matrices)
        probs_quant = get_inference_probs(fp16_sess, tensor_fp16, safe_matrices, threat_matrices)

        for name in threat_matrices.keys():
            orig_p = probs_orig[name]
            quant_p = probs_quant[name]
            drifts.append(abs(orig_p - quant_p))

            decision_orig = orig_p >= DriftConfig.THRESHOLD
            decision_quant = quant_p >= DriftConfig.THRESHOLD
            
            if decision_orig != decision_quant:
                flipped_cases.append({
                    "image": img_name,
                    "category": name,
                    "orig_prob": orig_p,
                    "quant_prob": quant_p,
                    "type": "Safe -> Threat (False Alarm)" if decision_quant else "Threat -> Safe (Missed)"
                })

    avg_drift = np.mean(drifts)
    print("\n" + "="*50)
    print(f"Final FP16 Quantization Stability Report:")
    print(f"Average Probability Drift: {avg_drift:.4f}%")
    print(f"Total Flipped Decisions: {len(flipped_cases)} out of {len(image_files) * len(threat_matrices)} checks")
    print("="*50)

    if flipped_cases:
        for case in flipped_cases[:5]:
            print(f"Image: {case['image']} | Cat: {case['category']} | {case['type']} | {case['orig_prob']:.2f}% -> {case['quant_prob']:.2f}%")
    else:
        print("\nSuccess: Absolute stability. Zero flipped decisions detected.")

if __name__ == "__main__":
    evaluate_quantization_drift()