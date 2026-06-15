"""
Module: flipped_decisions_test.py
Description: 
    Evaluates the robustness of the FP16 post-training quantization.
    It runs parallel pairwise inference using both the original FP32 Vision model 
    and the quantized FP16 Vision model, calculating the exact probability drift 
    and detecting if any safety thresholds were compromised (Flipped Decisions).
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor
from tqdm import tqdm

class EvalConfig:
    TEST_IMAGES_DIR = "test_images_sample" # Directory containing sample images
    NUM_IMAGES_TO_TEST = 100 
    
    ORIGINAL_MODELS = {"text": "../model_training/text_model.onnx", "vision": "../model_training/vision_model.onnx"}
    QUANTIZED_MODELS = {"text": "../model_training/text_model.onnx", "vision": "../model_training/vision_model_fp16.onnx"}

    SAFE_BASELINE = "Safe content, normal objects, everyday activities."
    THREAT_POLICIES = {
        "Violence": "Graphic physical violence, blood, weapons, combat.",
        "Adult": "Adult content, explicit clothing, nudity."
    }

    PROMPTS = [SAFE_BASELINE] + list(THREAT_POLICIES.values())
    THREAT_NAMES = list(THREAT_POLICIES.keys())
    THRESHOLD = 0.40


def get_pairwise_inference(session_dict, processor, image_path):
    """
    Executes a single inference pass and calculates pairwise probability scaling.
    Dynamically adjusts input types based on the ONNX graph's expected input type (FP32 vs FP16).
    """
    expected_type = session_dict['vision'].get_inputs()[0].type 
    target_np_type = np.float16 if "float16" in expected_type else np.float32

    text_embeds_list = []
    for prompt in EvalConfig.PROMPTS:
        t_in = processor(text=[prompt], return_tensors="np", padding="max_length", max_length=77, truncation=True)
        feed = {
            "input_ids": t_in['input_ids'].astype(np.int64),
            "attention_mask": t_in['attention_mask'].astype(np.int64)
        }
        out = session_dict['text'].run(["text_embeds"], feed)[0]
        out_f32 = out.astype(np.float32)
        text_embeds_list.append(out_f32 / np.linalg.norm(out_f32))
    text_embeds = np.vstack(text_embeds_list)

    # Vision Embedding Extraction
    image = Image.open(image_path).convert("RGB")
    img_input = processor(images=image, return_tensors="np")
    feed_img = {"pixel_values": img_input['pixel_values'].astype(target_np_type)}
    
    img_out = session_dict['vision'].run(["image_embeds"], feed_img)[0]
    img_embed = img_out.astype(np.float32) 
    img_embed = img_embed / np.linalg.norm(img_embed)

    LOGIT_SCALE = np.exp(4.60517)
    safe_logit = (img_embed @ text_embeds[0:1].T) * LOGIT_SCALE
    threat_logits = (img_embed @ text_embeds[1:].T) * LOGIT_SCALE
    
    threat_probs = []
    for i in range(len(EvalConfig.THREAT_NAMES)):
        t_log, s_log = threat_logits[0, i], safe_logit[0, 0]
        max_log = max(t_log, s_log)
        prob = np.exp(t_log - max_log) / (np.exp(t_log - max_log) + np.exp(s_log - max_log))
        threat_probs.append(prob)
        
    return np.array(threat_probs)


def compare_with_flipped_detection():
    """
    Iterates through the test dataset, computes inference on both FP32 and FP16 models,
    and logs any discrepancies or flipped decisions caused by precision loss.
    """
    if not os.path.exists(EvalConfig.TEST_IMAGES_DIR):
        print(f"Error: Test directory '{EvalConfig.TEST_IMAGES_DIR}' not found.")
        return

    print("Loading processor and initializing ONNX sessions...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    orig_sessions = {
        "text": ort.InferenceSession(EvalConfig.ORIGINAL_MODELS["text"], providers=['CPUExecutionProvider']), 
        "vision": ort.InferenceSession(EvalConfig.ORIGINAL_MODELS["vision"], providers=['CPUExecutionProvider'])
    }
    quant_sessions = {
        "text": ort.InferenceSession(EvalConfig.QUANTIZED_MODELS["text"], providers=['CPUExecutionProvider']), 
        "vision": ort.InferenceSession(EvalConfig.QUANTIZED_MODELS["vision"], providers=['CPUExecutionProvider'])
    }

    image_files = [f for f in os.listdir(EvalConfig.TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:EvalConfig.NUM_IMAGES_TO_TEST]
    
    if not image_files:
        print("No images found in the test directory.")
        return

    flipped_cases = []
    drifts = []

    print(f"Analyzing {len(image_files)} images for Quantization Drift...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(EvalConfig.TEST_IMAGES_DIR, img_name)
        
        probs_orig = get_pairwise_inference(orig_sessions, processor, img_path)
        probs_quant = get_pairwise_inference(quant_sessions, processor, img_path)
        
        drifts.append(np.abs(probs_orig - probs_quant))

        for i, name in enumerate(EvalConfig.THREAT_NAMES):
            decision_orig = probs_orig[i] >= EvalConfig.THRESHOLD
            decision_quant = probs_quant[i] >= EvalConfig.THRESHOLD
            
            if decision_orig != decision_quant:
                flipped_cases.append({
                    "image": img_name,
                    "category": name,
                    "orig_prob": probs_orig[i],
                    "quant_prob": probs_quant[i],
                    "type": "Safe -> Threat (False Alarm)" if decision_quant else "Threat -> Safe (Missed)"
                })

    avg_drift = np.mean(drifts) * 100
    
    print("\n" + "="*50)
    print(f"Final Quantization Stability Report:")
    print(f"Average Probability Drift: {avg_drift:.4f}%")
    print(f"Total Flipped Decisions: {len(flipped_cases)} out of {len(image_files) * len(EvalConfig.THREAT_NAMES)} checks")
    print("="*50)

    if flipped_cases:
        print("\nAlert: Flipped Cases Detected:")
        for case in flipped_cases[:5]:
            print(f"Image: {case['image']} | Category: {case['category']}")
            print(f"Drift Type: {case['type']} | Original: {case['orig_prob']:.4f} -> Quantized: {case['quant_prob']:.4f}\n")
    else:
        print("\nSuccess: Absolute stability. Zero flipped decisions detected.")

if __name__ == "__main__":
    compare_with_flipped_detection()