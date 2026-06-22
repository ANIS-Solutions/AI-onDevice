"""
Module: generate_baselines_onnx.py
Description:
    Generates baseline text embeddings for each policy category using the ONNX text encoder.
    These embeddings are stored in a JSON file and will be used for similarity comparisons during inference.
    
    IMPORTANT NOTE: you can easily switch between different policy versions by changing the POLICIES variable above.
    or create your own by following the same structure.
    Just ensure that the keys are unique category names and the values are lists of descriptive sentences that capture the essence of each category.
    The script will automatically generate embeddings for all provided categories and store them in the output JSON file for use during inference.

"""
import json
import numpy as np
import onnxruntime as ort
from transformers import CLIPProcessor

ONNX_MODEL_PATH = "text_model_single.onnx"
TOKENIZER_ID = "clip_local/" 
OUTPUT_JSON = "baselines.json"

POLICIES = {
    "SAFE_BASELINE": [
        "Safe, innocent, child-friendly content",
        "A child reading an educational book or doing homework",
        "A kid watching a safe, educational cartoon",
        "Everyday objects, vehicles, buildings, or UI menus"
    ],
    "THREAT_VIOLENCE": [
        "Explicit physical violence, blood, horror movies",
        "People fighting, punching, or holding weapons like guns or knives"
    ],
    "THREAT_ADULT": [
        "Content containing sexual revealing clothing, babydoll, lingerie",
        "Nudity or inappropriate adult romantic scenes"
    ],
    "THREAT_TEXT": [
        "Screenshots of chat messages containing explicit sexual text",
        "Text messages containing violent threats or cyberbullying"
    ]
}
# POLICIES_2 = {
#     "SAFE_BASELINE": ["Safe, innocent, everyday objects, normal content, people smiling."]

# ,
# "Adult_THREATS" :["adult themes, or Sexual revealing clothing (babydoll or lingerie) or Content that is LGBTQ+ friendly, short or tight dresses"],
# "Violence_THREATS": [
#      "Graphic physical violence, aggressive combat and fighting, people punching or attacking, visible blood and gore, severe injuries, individuals aiming firearms, holding sharp knives or dangerous weapons, disturbing horror imagery, warfare, and physical abuse."]
# }
# POLICIES_3 = {
#     "SAFE_GENERAL": [
#         "Safe and neutral content, everyday objects, normal casual clothing, standard vehicles, city streets, landscapes, people smiling and walking."
#     ],
    
#     "SAFE_EDUCATIONAL_ACTIVITY": [
#         "Positive educational activities, children studying, doing homework, playing outdoor sports, reading books, family bonding, solving logic puzzles."
#     ],
    
#     "THREAT_VIOLENCE": [
#         "Graphic physical violence, aggressive combat and fighting, severe physical injuries, individuals aiming firearms or weapons, visible blood, warfare, disturbing horror imagery."
#     ],
    
#     "THREAT_ADULT": [
#         "Adult themes, sexual revealing clothing, lingerie, tight dresses, partial or full nudity, inappropriate adult romantic or sexual scenes."
#     ],
    
#     "THREAT_TOXIC_TEXT": [
#         "Screenshots of text messages containing violent threats, severe insults, cyberbullying, aggressive hate speech, or explicit sexual language."
#     ]
# }
def generate_baselines_onnx():
    print("[INFO] Loading Tokenizer and ONNX Runtime Session...")
    processor = CLIPProcessor.from_pretrained(TOKENIZER_ID, force_download=False)
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

    output_data = {}

    for category, descriptions in POLICIES.items():
        print(f"[INFO] Processing category: {category}...")
        output_data[category] = {}
        
        for desc in descriptions:
            inputs = processor(
                text=[desc], 
                return_tensors="np", 
                padding="max_length", 
                truncation=True, 
                max_length=77
            )
            
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            ort_outputs = ort_session.run(None, ort_inputs)
            text_embeds = ort_outputs[0]
            
            norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
            text_embeds_normalized = text_embeds / norms
            
            output_data[category][desc] = text_embeds_normalized[0].tolist()

    print(f"[INFO] Saving ONNX-generated embeddings to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, indent=4)
        
    print("[INFO] Success! baselines.json generated using ONNX.")

if __name__ == "__main__":
    generate_baselines_onnx()