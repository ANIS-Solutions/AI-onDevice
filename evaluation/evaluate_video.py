"""
Module: evaluate_video.py
Description: 
    Executes real-time video inference using the hybrid asymmetric pipeline.
    Utilizes FP32 for static Text Embeddings and FP16 for dynamic Vision Inference.
    Applies Pairwise Probability Scaling and a Temporal Finite State Machine (FSM).
"""

import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
from transformers import CLIPProcessor
from tqdm import tqdm

class InferenceConfig:
    TEXT_ONNX_PATH = "text_model.onnx"
    VISION_ONNX_PATH = "vision_model_fp16.onnx"
    BASE_MODEL = "openai/clip-vit-base-patch32"
    INPUT_VIDEO = "test_input.mp4"
    OUTPUT_VIDEO = "test_output.mp4"

    SAFE_BASELINE = "Safe, innocent, everyday objects, normal content, people smiling."
    ACTIVE_THREATS = {
        "Adult": "adult themes, sexual revealing clothing, lingerie, tight dresses",
        "Violence": "Graphic physical violence, aggressive combat, blood, weapons, physical abuse",
    }
    
    THRESHOLDS = {"Violence": 0.45, "Adult": 0.40}
    TRIGGER_DELAY_MS = 400 
    RELEASE_DELAY_MS = 3000


def run_hybrid_inference():
    """Runs the full evaluation pipeline on a local video file."""
    print("Loading ONNX sessions...")
    providers = ['CPUExecutionProvider']
    text_session = ort.InferenceSession(InferenceConfig.TEXT_ONNX_PATH, providers=providers)
    vision_session = ort.InferenceSession(InferenceConfig.VISION_ONNX_PATH, providers=providers)
    processor = CLIPProcessor.from_pretrained(InferenceConfig.BASE_MODEL)

    prompts = [InferenceConfig.SAFE_BASELINE] + list(InferenceConfig.ACTIVE_THREATS.values())
    threat_names = list(InferenceConfig.ACTIVE_THREATS.keys())

    print("Pre-computing Text Embeddings...")
    text_embeds_list = []
    for prompt in prompts:
        t_in = processor(text=[prompt], return_tensors="np", padding="max_length", max_length=77, truncation=True)
        feed = {
            "input_ids": t_in['input_ids'].astype(np.int64),
            "attention_mask": t_in['attention_mask'].astype(np.int64)
        }
        out = text_session.run(None, feed)[0]
        out_f32 = out.astype(np.float32)
        text_embeds_list.append(out_f32 / np.linalg.norm(out_f32))

    text_features = np.vstack(text_embeds_list)
    safe_feature = text_features[0:1]
    threat_features = text_features[1:]

    print("Processing video stream...")
    cap = cv2.VideoCapture(InferenceConfig.INPUT_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_video = cv2.VideoWriter(InferenceConfig.OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    trigger_limit = int((InferenceConfig.TRIGGER_DELAY_MS / 1000.0) * fps)
    release_limit = int((InferenceConfig.RELEASE_DELAY_MS / 1000.0) * fps)

    system_state = "SAFE"
    state_counter = 0
    probs_text = ""

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: 
            break

        if frame_idx % max(1, int(fps/3)) == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_in = processor(images=pil_img, return_tensors="np")['pixel_values']
            
            feed_img = {"pixel_values": img_in.astype(np.float16)}
            img_out = vision_session.run(None, feed_img)[0]
            
            img_feat = img_out.astype(np.float32)
            img_feat = img_feat / np.linalg.norm(img_feat)

            scale = np.exp(4.60517)
            s_log = (img_feat @ safe_feature.T) * scale
            t_logs = (img_feat @ threat_features.T) * scale
            
            current_threat = False
            parts = []
            
            for i, name in enumerate(threat_names):
                t_log, sl = t_logs[0, i], s_log[0, 0]
                m = max(t_log, sl)
                prob = np.exp(t_log - m) / (np.exp(t_log - m) + np.exp(sl - m))
                parts.append(f"{name[:3]}:{prob*100:.0f}%")
                
                if prob >= InferenceConfig.THRESHOLDS.get(name, 0.5):
                    current_threat = True
                    
            probs_text = " | ".join(parts)

        # Finite State Machine (FSM) Logic
        if current_threat:
            if system_state == "SAFE": 
                system_state, state_counter = "PENDING_BLUR", 1
            elif system_state == "PENDING_BLUR":
                state_counter += 1
                if state_counter >= trigger_limit: system_state = "BLURRED"
            elif system_state == "PENDING_RELEASE": 
                system_state, state_counter = "BLURRED", 0
        else:
            if system_state == "BLURRED": 
                system_state, state_counter = "PENDING_RELEASE", 1
            elif system_state == "PENDING_RELEASE":
                state_counter += 1
                if state_counter >= release_limit: system_state = "SAFE"
            elif system_state == "PENDING_BLUR": 
                system_state, state_counter = "SAFE", 0

        # UI Rendering
        display = frame.copy()
        if system_state in ["BLURRED", "PENDING_RELEASE"]:
            display = cv2.GaussianBlur(display, (99, 99), 0)
        
        cv2.rectangle(display, (10, 10), (600, 95), (0,0,0), -1)
        color = (0, 255, 0) if system_state == "SAFE" else (0, 0, 255)
        cv2.putText(display, f"STATE: {system_state}", (20, 45), 1, 1.8, color, 2)
        cv2.putText(display, probs_text, (20, 85), 1, 1.3, (255, 255, 255), 1)
        
        out_video.write(display)

    cap.release()
    out_video.write(display) # ensure flush
    out_video.release()
    print("Inference completed successfully.")

if __name__ == "__main__":
    run_hybrid_inference()