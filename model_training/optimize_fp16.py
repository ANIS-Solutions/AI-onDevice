"""
Module: optimize_fp16.py
Description: 
    Converts exported ONNX models from FP32 (Float32) to FP16 (Float16) 
    to halve the memory footprint for edge device execution.
"""

import onnx
from onnxconverter_common import float16
import os

def convert_to_fp16(input_path, output_path, model_name):
    """Converts a specific ONNX model to FP16 format."""
    if not os.path.exists(input_path):
        print(f"Error: Source file {input_path} not found.")
        return

    print(f"Converting {model_name} to Float16...")
    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model)
    
    onnx.save(model_fp16, output_path)
    
    size_before = os.path.getsize(input_path) / (1024 * 1024)
    size_after = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Conversion successful. {model_name} saved to {output_path}")
    print(f"Size Reduction: {size_before:.2f} MB -> {size_after:.2f} MB\n")

if __name__ == "__main__":
    # Usually, only the Vision model is converted to FP16 for edge inference
    convert_to_fp16("vision_model.onnx", "vision_model_fp16.onnx", "Vision Model")