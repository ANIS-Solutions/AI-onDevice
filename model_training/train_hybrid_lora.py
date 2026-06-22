"""
Module: train_hybrid_lora.py
Description: 
    Multi-Task Learning for CLIP using LoRA.
    Task 1: Contrastive Loss (InfoNCE) using rich 'clip_description' to learn deep nuances.
    Task 2: Classification Loss (BCEWithLogitsLoss) using 'final_label' mapped to static prompts.
"""

import os
import gc
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

class Config:
    MODEL_ID = "openai/clip-vit-base-patch32"
    DATA_PATH = "final_dataset_v3_ready.csv"  # Ensure this has: filename, clip_description, final_label
    IMG_DIR = "train/"
    OUTPUT_DIR = "clip_hybrid_best"
    
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    WEIGHT_CONTRASTIVE = 0.5
    WEIGHT_BCE = 0.5

    LABELS_DICT = {
        "Safe Content": "Safe, innocent, child-friendly content or movie, sports theme, normal or casual clothes.",
        "Neutral Objects": "Everyday objects, cars, vehicles, buildings, blank walls, street views, UI menus.",
        "Violence": "Explicit physical violence, blood, horror movies, people fighting, holding weapons.",
        "Inappropriate Text": "Screenshots of chat messages containing explicit sexual text or violent threats.",
        "Adult Content": "Content containing sexual revealing clothing, babydoll, lingerie, tight dresses."
    }
    CATEGORY_NAMES = list(LABELS_DICT.keys())
    STATIC_PROMPTS = list(LABELS_DICT.values())
    NUM_CLASSES = len(CATEGORY_NAMES)


class HybridCLIPDataset(Dataset):
    def __init__(self, df, processor, img_folder):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_folder, str(row['filename']))
        rich_description = str(row['clip_description'])
        actual_label = row['final_label']
        
        label_vector = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        if actual_label in ["Safe_General", "Safe_Contextual_Body"]: 
            label_vector[0] = 1.0
        elif actual_label == "Unsafe_Violence": 
            label_vector[2] = 1.0
        elif actual_label == "Unsafe_Text": 
            label_vector[3] = 1.0
        elif actual_label == "Unsafe_Sexual": 
            label_vector[4] = 1.0
        else: 
            label_vector[1] = 1.0

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')

        image_inputs = self.processor(images=image, return_tensors="pt")
        
        text_inputs = self.processor(
            text=rich_description, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        )

        return {
            "pixel_values": image_inputs['pixel_values'].squeeze(0),
            "rich_input_ids": text_inputs['input_ids'].squeeze(0),
            "rich_attention_mask": text_inputs['attention_mask'].squeeze(0),
            "label_vector": label_vector
        }


def train_hybrid_model():
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Initializing base model and processor...")
    model = CLIPModel.from_pretrained(Config.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(Config.MODEL_ID)
    model.to(Config.DEVICE)

    # Co-Tuning both Encoders
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    df = pd.read_csv(Config.DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = HybridCLIPDataset(train_df, processor, Config.IMG_DIR)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = nn.CrossEntropyLoss() # Used for InfoNCE

    # Pre-tokenize Static Prompts (Tensors stay fixed, but embeddings update inside loop)
    static_text_inputs = processor(
        text=Config.STATIC_PROMPTS, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=77
    ).to(Config.DEVICE)

    print("Starting Hybrid Training Loop...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(Config.DEVICE)
            rich_input_ids = batch["rich_input_ids"].to(Config.DEVICE)
            rich_attention_mask = batch["rich_attention_mask"].to(Config.DEVICE)
            labels = batch["label_vector"].to(Config.DEVICE) 
            
            optimizer.zero_grad()

            
            vision_outputs = model.base_model.model.vision_model(pixel_values=pixel_values)
            img_pooled = vision_outputs[1] # Extract pooler_output
            img_features = model.base_model.model.visual_projection(img_pooled)
            img_features = img_features / img_features.norm(dim=1, keepdim=True)
            
            rich_outputs = model.base_model.model.text_model(
                input_ids=rich_input_ids, 
                attention_mask=rich_attention_mask
            )
            rich_pooled = rich_outputs[1]
            rich_text_features = model.base_model.model.text_projection(rich_pooled)
            rich_text_features = rich_text_features / rich_text_features.norm(dim=1, keepdim=True)

            static_outputs = model.base_model.model.text_model(
                input_ids=static_text_inputs['input_ids'], 
                attention_mask=static_text_inputs['attention_mask']
            )
            static_pooled = static_outputs[1]
            static_text_features = model.base_model.model.text_projection(static_pooled)
            static_text_features = static_text_features / static_text_features.norm(dim=1, keepdim=True)
            
            logit_scale = model.base_model.model.logit_scale.exp()
            logits_per_image = logit_scale * img_features @ rich_text_features.T
            logits_per_text = logits_per_image.T
            ground_truth = torch.arange(len(pixel_values), dtype=torch.long, device=Config.DEVICE)
            
            loss_img = contrastive_criterion(logits_per_image, ground_truth)
            loss_txt = contrastive_criterion(logits_per_text, ground_truth)
            contrastive_loss = (loss_img + loss_txt) / 2

            logits_static = logit_scale * img_features @ static_text_features.T 
            bce_loss = bce_criterion(logits_static, labels)

            loss = (Config.WEIGHT_CONTRASTIVE * contrastive_loss) + (Config.WEIGHT_BCE * bce_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                "Total": f"{loss.item():.4f}", 
                "Contr": f"{contrastive_loss.item():.4f}", 
                "BCE": f"{bce_loss.item():.4f}"
            })
            
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader):.4f}")
        
        model.save_pretrained(Config.OUTPUT_DIR)
        processor.save_pretrained(Config.OUTPUT_DIR)

if __name__ == "__main__":
    train_hybrid_model()