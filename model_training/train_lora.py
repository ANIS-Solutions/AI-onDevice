"""
Module: train_lora.py
Description: 
    Fine-tunes the CLIP Vision-Language Model using Low-Rank Adaptation (LoRA).
    Implements a Multi-Label classification approach utilizing BCEWithLogitsLoss
    and independent Sigmoid activations to prevent probability dilution.
"""

import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

class Config:
    """Configuration parameters for the training pipeline."""
    MODEL_ID = "openai/clip-vit-base-patch32"
    DATA_PATH = "final_dataset_v3_ready.csv"
    IMG_DIR = "train/"
    OUTPUT_DIR = "clip_v2_sigmoid_best"
    
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LABELS_DICT = {
        "Safe Content": "Safe, innocent, child-friendly content or movie, sports theme, normal or casual clothes.",
        "Neutral Objects": "Everyday objects, cars, vehicles, buildings, blank walls, street views, UI menus.",
        "Violence": "Explicit physical violence, blood, horror movies, people fighting, holding weapons.",
        "Inappropriate Text": "Screenshots of chat messages containing explicit sexual text or violent threats.",
        "Adult Content": "Content containing sexual revealing clothing, babydoll, lingerie, tight dresses."
    }
    CATEGORY_NAMES = list(LABELS_DICT.keys())
    PROMPTS = list(LABELS_DICT.values())
    NUM_CLASSES = len(CATEGORY_NAMES)


class MultiLabelCLIPDataset(Dataset):
    """Custom Dataset for loading images and generating multi-hot label vectors."""
    def __init__(self, df, processor, img_folder):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_folder, row['filename'])
        
        actual_label = row['final_label']
        label_vector = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        
        if actual_label in ["Safe_General", "Safe_Contextual_Body"]: 
            label_vector[0] = 1.0
        elif actual_label == "Unsafe_Violence": 
            label_vector[2] = 1.0
        elif actual_label == "Unsafe_Sexual": 
            label_vector[4] = 1.0
        else: 
            label_vector[1] = 1.0

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')

        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "label_vector": label_vector
        }


def train_model():
    """Executes the LoRA fine-tuning process."""
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Initializing model and processor...")
    model = CLIPModel.from_pretrained(Config.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(Config.MODEL_ID)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.to(Config.DEVICE)
    
    print("Preparing dataset...")
    df = pd.read_csv(Config.DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = MultiLabelCLIPDataset(train_df, processor, Config.IMG_DIR)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    text_inputs = processor(text=Config.PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=77).to(Config.DEVICE)

    print("Starting training loop...")
    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(Config.DEVICE)
            labels = batch["label_vector"].to(Config.DEVICE) 
            
            optimizer.zero_grad()

            dummy_pixels = torch.zeros((text_inputs['input_ids'].shape[0], 3, 224, 224)).to(Config.DEVICE)
            text_outputs = model(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], pixel_values=dummy_pixels)
            text_features = text_outputs.text_embeds
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            dummy_ids = torch.zeros((pixel_values.shape[0], 77), dtype=torch.long).to(Config.DEVICE)
            dummy_mask = torch.zeros((pixel_values.shape[0], 77), dtype=torch.long).to(Config.DEVICE)
            image_outputs = model(input_ids=dummy_ids, attention_mask=dummy_mask, pixel_values=pixel_values)
            img_features = image_outputs.image_embeds
            img_features = img_features / img_features.norm(dim=1, keepdim=True)
            
            logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else model.base_model.model.logit_scale.exp()
            logits = logit_scale * img_features @ text_features.T 
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader):.4f}")
        model.save_pretrained(Config.OUTPUT_DIR)
        processor.save_pretrained(Config.OUTPUT_DIR)

if __name__ == "__main__":
    train_model()