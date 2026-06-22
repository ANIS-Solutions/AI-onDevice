"""
Module: evaluate_video.py
Description: 
    Real-time video inference. Uses the dynamic JSON policy instead of a Text Model.
    Features a 2-FPS Steady Heartbeat and an FSM for flicker prevention.
"""

import cv2
import json
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

class VideoConfig:
    VISION_ONNX_PATH = "../model_training/vision_model_fp16.onnx"
    POLICY_JSON = "baselines.json"
    INPUT_VIDEO = "test_input.mp4"
    OUTPUT_VIDEO = "test_output_dynamic.mp4"
    
    TARGET_FPS = 2
    LOGIT_SCALE = 100.0
    DEFAULT_THRESHOLD = 65.0
    
    # FSM Variables (in ticks, assuming TARGET_FPS=2)
    TRIGGER_TICKS = 2  # 1 second of threat
    RELEASE_TICKS = 3  # 1.5 seconds of safety

def run_dynamic_video_inference():
    with open(VideoConfig.POLICY_JSON, 'r', encoding='utf-8') as f:
        policy = json.load(f)

    safe_matrices = [np.array(list(d.values()), dtype=np.float32).T for k, d in policy.items() if "SAFE" in k.upper()]
    threat_matrices = {k: np.array(list(d.values()), dtype=np.float32).T for k, d in policy.items() if "SAFE" not in k.upper()}

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    vision_session = ort.InferenceSession(VideoConfig.VISION_ONNX_PATH, providers=providers)
    input_name = vision_session.get_inputs()[0].name

    def preprocess(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        mean, std = np.array([0.481, 0.457, 0.408]), np.array([0.268, 0.261, 0.275])
        img = (img - mean) / std
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float16)

    cap = cv2.VideoCapture(VideoConfig.INPUT_VIDEO)
    if not cap.isOpened(): return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_video = cv2.VideoWriter(VideoConfig.OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_interval = max(1, int(fps / VideoConfig.TARGET_FPS))
    
    system_state = "SAFE"
    trigger_counter, release_counter = 0, 0
    last_probs = {k: 0.0 for k in threat_matrices.keys()}

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        run_inference = (frame_idx % frame_interval == 0)

        if run_inference:
            tensor = preprocess(frame)
            out = vision_session.run(None, {input_name: tensor})[0]
            img_embed = out[0] / np.linalg.norm(out[0])
            
            safe_sims = [np.max(np.dot(img_embed, mat)) for mat in safe_matrices if mat.shape[0] > 0]
            best_safe_sim = max(safe_sims) if safe_sims else -1.0
            
            is_unsafe_now = False
            for t_name, t_mat in threat_matrices.items():
                sim_threat = np.max(np.dot(img_embed, t_mat))
                diff = np.clip((sim_threat - best_safe_sim) * VideoConfig.LOGIT_SCALE, -50, 50)
                prob = (1.0 / (1.0 + np.exp(-diff))) * 100.0
                last_probs[t_name] = prob
                if prob >= VideoConfig.DEFAULT_THRESHOLD:
                    is_unsafe_now = True

            # FSM Logic
            if is_unsafe_now:
                release_counter = 0
                trigger_counter += 1
                if trigger_counter >= VideoConfig.TRIGGER_TICKS:
                    system_state = "BLOCKED"
            else:
                trigger_counter = 0
                release_counter += 1
                if release_counter >= VideoConfig.RELEASE_TICKS:
                    system_state = "SAFE"

        # UI Rendering
        display = frame.copy()
        if system_state == "BLOCKED":
            cv2.rectangle(display, (0, 0), (width, height), (0, 0, 200), -1)
            cv2.addWeighted(display, 0.4, frame, 0.6, 0, display)
            color, state_txt = (0, 0, 255), "BLOCKED"
        else:
            color, state_txt = (0, 255, 0), "SAFE"

        cv2.putText(display, f"STATE: {state_txt}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        y_offset = 100
        for category, prob in last_probs.items():
            txt_color = (0, 0, 255) if prob >= VideoConfig.DEFAULT_THRESHOLD else (255, 255, 255)
            cv2.putText(display, f"{category}: {prob:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)
            y_offset += 35

        out_video.write(display)

    cap.release()
    out_video.release()
    print("Inference completed successfully.")

if __name__ == "__main__":
    run_dynamic_video_inference()