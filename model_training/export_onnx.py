"""
Module: export_onnx.py
Description: 
    Splits a merged CLIP model into independent Text and Vision encoders.
    Exports both modalities to the ONNX format (opset 14) for edge deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPProcessor

MERGED_DIR = "merged_clip_model"

class TextEncoderWrapper(nn.Module):
    """Wrapper to force clean ONNX export without dictionary outputs."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]


class VisionEncoderWrapper(nn.Module):
    """Wrapper to force clean ONNX export without dictionary outputs."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, return_dict=False)[0]


def export_models():
    """Loads specialized encoders and exports them to ONNX."""
    print("Loading specialized Text and Vision models...")
    text_model = CLIPTextModelWithProjection.from_pretrained(MERGED_DIR)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(MERGED_DIR)
    processor = CLIPProcessor.from_pretrained(MERGED_DIR)

    text_onnx = TextEncoderWrapper(text_model).eval()
    vision_onnx = VisionEncoderWrapper(vision_model).eval()

    print("Preparing dummy tensors for tracing...")
    dummy_text = processor(text=["dummy"], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    dummy_img_input = processor(images=dummy_image, return_tensors="pt")

    print("Exporting TextEncoder to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            text_onnx,
            (dummy_text['input_ids'], dummy_text['attention_mask']),
            "text_model.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_embeds']
        )

    print("Exporting VisionEncoder to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            vision_onnx,
            (dummy_img_input['pixel_values'],),
            "vision_model.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['image_embeds']
        )
        
    print("Export complete. Models saved as 'text_model.onnx' and 'vision_model.onnx'.")

if __name__ == "__main__":
    export_models()