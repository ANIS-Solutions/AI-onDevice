'''
Module: quantize_onnx.py
Description: 
    Post-training quantization pipeline for CLIP encoders.
    1. Load merged LoRA model and separate text/vision encoders.
    2. Export both encoders to ONNX (FP32).
    3. Quantize vision encoder to FP16 for mobile deployment.
    4. Pack text encoder into a single ONNX file for server use.
'''
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from peft import PeftModel
import onnx
from onnxconverter_common import float16

BASE_MODEL_ID = "openai/clip-vit-base-patch32"
LORA_DIR = "clip_hybrid_best"
MERGED_DIR = "merged_clip_model"

class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

class VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, return_dict=False)[0]

def run_pipeline():
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)

    print("[INFO] Loading and merging LoRA weights...")
    base_model = CLIPModel.from_pretrained(BASE_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_ID)
    
    peft_model = PeftModel.from_pretrained(base_model, LORA_DIR)
    merged_model = peft_model.merge_and_unload()
    
    merged_model.save_pretrained(MERGED_DIR)
    processor.save_pretrained(MERGED_DIR)
    print("[INFO] Merged model saved.")

    print("[INFO] Loading separated encoders...")
    text_model = CLIPTextModelWithProjection.from_pretrained(MERGED_DIR)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(MERGED_DIR)
    
    text_wrapper = TextEncoderWrapper(text_model).eval()
    vision_wrapper = VisionEncoderWrapper(vision_model).eval()

    dummy_text = processor(text=["dummy"], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    dummy_img_input = processor(images=dummy_image, return_tensors="pt")

    print("[INFO] Exporting Text Encoder (FP32)...")
    with torch.no_grad():
        torch.onnx.export(
            text_wrapper,
            (dummy_text['input_ids'], dummy_text['attention_mask']),
            "text_model_fp32.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_embeds']
        )

    print("[INFO] Exporting Vision Encoder (FP32)...")
    with torch.no_grad():
        torch.onnx.export(
            vision_wrapper,
            (dummy_img_input['pixel_values'],),
            "vision_model_fp32.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['image_embeds']
        )

    print("[INFO] Quantizing Vision Encoder to FP16...")
    vision_fp32 = onnx.load("vision_model_fp32.onnx")
    vision_fp16 = float16.convert_float_to_float16(vision_fp32)
    onnx.save(vision_fp16, "vision_model_fp16.onnx")

    print("[INFO] Packing Text Encoder into a single file...")
    text_model_packed = onnx.load("text_model_fp32.onnx", load_external_data=True)
    onnx.save(text_model_packed, "text_model_single.onnx")

    print("[INFO] Cleaning up temporary files...")
    temp_files = [
        "vision_model_fp32.onnx", 
        "vision_model_fp32.onnx.data",
        "text_model_fp32.onnx",
        "text_model_fp32.onnx.data"
    ]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)

    print("\n[SUCCESS] Pipeline finished! Your final production models are:")
    print(" Mobile: vision_model_fp16.onnx")
    print("Server: text_model_single.onnx")

if __name__ == "__main__":
    run_pipeline()