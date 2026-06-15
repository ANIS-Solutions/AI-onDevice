"""
Module: generate_predictions.py
Description: 
    Executes the ONNX model inference across the test datasets 
    (Static Images, Kinetics, UCF Crime) and saves raw probability outputs to CSV files.
"""

import os
import cv2
import json
import glob
import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm

class EvalConfig:
    VISION_ONNX_PATH = "../model_training/vision_model_fp16.onnx"
    JSON_PATH = "saved_embeddings.json"
    BASE_DIR = "test"
    TARGET_FPS = 2
    LOGIT_SCALE = 100.0
    DEFAULT_THRESHOLD = 0.85

class EdgeModerator:
    def __init__(self):
        with open(EvalConfig.JSON_PATH, 'r', encoding='utf-8') as f:
            edge_data = json.load(f)

        self.safe_baseline = np.array(edge_data['Baseline'], dtype=np.float32)
        self.safe_baseline /= np.linalg.norm(self.safe_baseline)

        self.threats = {}
        for t_name, t_emb_list in edge_data.items():
            if t_name != 'Baseline':
                t_emb = np.array(t_emb_list, dtype=np.float32)
                self.threats[t_name] = t_emb / np.linalg.norm(t_emb)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(EvalConfig.VISION_ONNX_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float16)

    def calculate_probabilities(self, emb):
        sim_safe = np.dot(emb, self.safe_baseline)
        probs = {}
        for t_name, t_emb in self.threats.items():
            sim_threat = np.dot(emb, t_emb)
            diff = np.clip((sim_threat - sim_safe) * EvalConfig.LOGIT_SCALE, -50, 50)
            probs[t_name] = 1.0 / (1.0 + np.exp(-diff))
        return probs

def process_image(moderator, img_path, ground_truth_label="Unknown"):
    frame = cv2.imread(img_path)
    if frame is None: return None
    
    tensor = moderator.preprocess(frame)
    out = moderator.session.run(None, {moderator.input_name: tensor})[0]
    emb = out[0] / np.linalg.norm(out[0])
    
    probs = moderator.calculate_probabilities(emb)
    is_unsafe = any(prob >= EvalConfig.DEFAULT_THRESHOLD for prob in probs.values())
        
    result = {
        "filename": os.path.basename(img_path), 
        "ground_truth_label": ground_truth_label,
        "predicted_status": "BLOCKED" if is_unsafe else "SAFE"
    }
    for t_name, prob in probs.items():
        result[f"{t_name}_prob"] = round(prob * 100, 2)
    return result

def run_inference_pipeline():
    moderator = EdgeModerator()
    results_images = []

    image_folder = os.path.join(EvalConfig.BASE_DIR, "old_test", "old_test_set")
    image_csv = os.path.join(EvalConfig.BASE_DIR, "old_test", "train_split.csv")
    
    if os.path.exists(image_folder) and os.path.exists(image_csv):
        print("Processing Static Images...")
        img_df = pd.read_csv(image_csv)
        img_label_dict = dict(zip(img_df['filename'], img_df['final_label']))
        
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            
        for img_path in tqdm(image_paths, desc="Images", unit="img"):
            ground_truth = img_label_dict.get(os.path.basename(img_path), "Unknown") 
            res = process_image(moderator, img_path, ground_truth)
            if res: results_images.append(res)
            
        pd.DataFrame(results_images).to_csv("results_images.csv", index=False)
        print("Raw predictions saved to results_images.csv")

if __name__ == "__main__":
    run_inference_pipeline()