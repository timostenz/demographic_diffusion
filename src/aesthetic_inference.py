import torch
import open_clip
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import torch.nn as nn
from os.path import expanduser
from urllib.request import urlretrieve
import torch.nn.functional as F
from tqdm import tqdm

# Load CLIP model
def load_clip_model(clip_model_name="ViT-L-14", device="cpu"):
    model = open_clip.create_model(clip_model_name, pretrained="openai")
    preprocess = open_clip.transform.image_transform(model.visual.image_size, is_train=False)
    model.to(device).eval()
    return model, preprocess

# Get image embeddings
def get_clip_embedding(image_url, model, preprocess, device="cpu"):

    try:
        # Download and preprocess the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except (requests.RequestException, IOError) as e:
        print(f"Error loading image: {e}")
        return None
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Extract embeddings
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).float()
        embedding = F.normalize(embedding, p=2, dim=-1)  # L2 Normalization

    return embedding


def get_clip_embedding_from_image(image, model, preprocess, device="cpu"):
    """Get CLIP embedding from a PIL image object."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).float()
        embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding

# function adapted from https://github.com/LAION-AI/aesthetic-predictor?tab=readme-ov-file
def get_aesthetic_model(clip_model="vit_l_14", device="cpu"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    
    # Define model based on CLIP type
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError("Unsupported CLIP model")
    
    # Load weights and move to device
    m.load_state_dict(torch.load(path_to_model, map_location=device))
    m.to(device).eval()
    return m

def get_aesthetic_score(image_url, clip_model, preprocess, aesthetic_model, device):
    """Return the aesthetic score and embedding for a given image URL."""
    embedding = get_clip_embedding(image_url, clip_model, preprocess, device)
    if embedding is None:
        return None, None  # Skip if image loading fails
    with torch.no_grad():
        score = aesthetic_model(embedding).item()
    return score, embedding.squeeze(0).cpu().numpy().tolist()

def process_image_column(df, column_name, clip_model_name="ViT-L-14", device=None, drop_invalid=True, batch_size=32, save_interval=20000, output_path=None, return_df=False):
    """Process a DataFrame column of image URLs, compute aesthetic scores and embeddings, and save results."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = load_clip_model(clip_model_name, device)
    aesthetic_model = get_aesthetic_model("vit_l_14", device)
    if "aesthetic_score" in df.columns:
        processed_rows = df["aesthetic_score"].notna().sum()
    else:
        df["aesthetic_score"] = None
        processed_rows = 0
    if "clip_embedding" not in df.columns:
        df["clip_embedding"] = None
    image_urls = df[column_name].tolist()
    for i in tqdm(range(processed_rows, len(image_urls), batch_size), desc="Processing Images (Batch Size 32)"):
        batch_urls = image_urls[i : i + batch_size]
        batch_scores = []
        batch_embeddings = []
        for url in batch_urls:
            score, embedding = get_aesthetic_score(url, clip_model, preprocess, aesthetic_model, device)
            batch_scores.append(score)
            batch_embeddings.append(embedding)
        df.loc[:, "aesthetic_score"].iloc[i:i+batch_size] = batch_scores
        df.loc[:, "clip_embedding"].iloc[i:i+batch_size] = batch_embeddings
        if output_path and i % save_interval == 0:
            df.to_parquet(output_path, index=False)
    if drop_invalid:
        df = df.dropna(subset=["aesthetic_score"]).reset_index(drop=True)
    if return_df:
        df.to_parquet(output_path, index=False)
        return df
    else:
        df.to_parquet(output_path, index=False)
        return