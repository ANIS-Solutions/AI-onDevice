"""
Module: generate_predictions.py
Description: 
    Generates dataset predictions based on the dynamic, user-defined JSON policy.
    Automatically adapts to any number of threat categories.
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
    VISION_ONNX_PATH = "vision_model_fp16.onnx"
    POLICY_JSON = "baselines_2.json"
    DATASET_DIR = "/mnt/d/projects/graduation_project/10/test/test/old_test/old_test_set"
    DATASET_CSV = "/mnt/d/projects/graduation_project/10/test/test/old_test/train_split.csv"
    LOGIT_SCALE = 100.0

class DynamicEdgeModerator:
    def __init__(self):
        with open(EvalConfig.POLICY_JSON, 'r', encoding='utf-8') as f:
            policy = json.load(f)

        self.safe_matrices = [np.array(list(data.values()), dtype=np.float32).T for k, data in policy.items() if "SAFE" in k.upper()]
        self.threat_matrices = {k: np.array(list(data.values()), dtype=np.float32).T for k, data in policy.items() if "SAFE" not in k.upper()}

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
        safe_sims = [np.max(np.dot(emb, mat)) for mat in self.safe_matrices if mat.shape[0] > 0]
        best_safe_sim = max(safe_sims) if safe_sims else -1.0

        probs = {}
        for t_name, t_mat in self.threat_matrices.items():
            sim_threat = np.max(np.dot(emb, t_mat))
            diff = np.clip((sim_threat - best_safe_sim) * EvalConfig.LOGIT_SCALE, -50, 50)
            probs[t_name] = (1.0 / (1.0 + np.exp(-diff))) * 100.0
        return probs

def process_image(moderator, img_path, ground_truth_label):
    frame = cv2.imread(img_path)
    if frame is None: return None
    
    tensor = moderator.preprocess(frame)
    out = moderator.session.run(None, {moderator.input_name: tensor})[0]
    emb = out[0] / np.linalg.norm(out[0])
    
    probs = moderator.calculate_probabilities(emb)
    
    result = {"filename": os.path.basename(img_path), "ground_truth_label": ground_truth_label}
    for t_name, prob in probs.items():
        result[f"{t_name}_prob"] = prob
    return result

def run_inference_pipeline():
    moderator = DynamicEdgeModerator()
    results = []

    image_folder = EvalConfig.DATASET_DIR
    
    if os.path.exists(image_folder) and os.path.exists(EvalConfig.DATASET_CSV):
        print(f"Processing Static Images against {len(moderator.threat_matrices)} custom threats...")
        img_df = pd.read_csv(EvalConfig.DATASET_CSV)
        img_label_dict = dict(zip(img_df['filename'], img_df['final_label']))
        
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            
        for img_path in tqdm(image_paths, desc="Evaluating Dataset"):
            ground_truth = img_label_dict.get(os.path.basename(img_path), "Unknown") 
            res = process_image(moderator, img_path, ground_truth)
            if res: results.append(res)
            
        pd.DataFrame(results).to_csv("results_dynamic_images.csv", index=False)
        print("Raw predictions saved to results_dynamic_images.csv")

if __name__ == "__main__":
    run_inference_pipeline()