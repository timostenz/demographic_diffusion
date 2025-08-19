import os
import pandas as pd
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from aesthetic_inference import load_clip_model, get_aesthetic_model, get_clip_embedding_from_image
from reward_function import ParquetDeepFMDatasetDirectEmbed, DeepFM


def load_pipeline(device="cuda"):
    """Load and return a Stable Diffusion pipeline on the specified device."""
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_images(prompts, pipe):
    """Generate images from a list of prompts using the provided pipeline."""
    images = []
    for prompt in tqdm(prompts, desc="Generating images"):
        with torch.no_grad():
            image = pipe(prompt).images[0]
            images.append(image)
    return images


def compute_embeddings(images, prompts, device="cuda"):
    """Compute CLIP image embeddings and aesthetic scores for a list of images."""
    clip_model, preprocess = load_clip_model("ViT-L-14", device)
    aesthetic_model = get_aesthetic_model("vit_l_14", device)
    image_embeddings = []
    aesthetic_scores = []
    for img, _ in tqdm(zip(images, prompts), total=len(prompts), desc="Computing embeddings"):
        image_embed = get_clip_embedding_from_image(img, clip_model, preprocess, device)
        score = aesthetic_model(image_embed).item() if image_embed is not None else None
        image_embeddings.append(image_embed.squeeze(0).cpu().numpy().tolist())
        aesthetic_scores.append(score)
    return image_embeddings, aesthetic_scores


def generate_images_embeddings(prompt_file, output_file):
    """Generate images and compute embeddings/scores for prompts in a file, then save results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(prompt_file) if prompt_file.endswith(".parquet") else pd.read_csv(prompt_file)
    prompts = df["prompt"].tolist()
    pipe = load_pipeline(device)
    images = generate_images(prompts, pipe)
    image_embeds, scores = compute_embeddings(images, prompts, device)
    df["clip_embedding"] = image_embeds
    df["aesthetic_score"] = scores
    # fixed scaling
    min_val = 0.21
    max_val = 8.37
    df["aesthetic_score"] = (df["aesthetic_score"] - min_val) / (max_val - min_val)
    df.to_parquet(output_file, index=False)
    print(f"Baseline data saved to {output_file}")

def compute_deepfm_reward(data_path, model_path, output_path):
    """Compute and save DeepFM model rewards for a dataset."""
    df = pd.read_parquet(data_path)
    feature_columns = [
        "aesthetic_score", "ratings", "discount_percentage",
        "discount_price_log", "actual_price_log"
    ]
    categorical_cols = ["main_category", "sub_category", "gender", "age_group"]
    dataset = ParquetDeepFMDatasetDirectEmbed(
        df=df,
        feature_columns=feature_columns,
        categorical_cols=categorical_cols,
        text_embed_col="clip_text_embedding",
        image_embed_col="clip_embedding"
    )
    model = load_deepfm_model(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    Xi, Xv = zip(*[dataset[i] for i in range(len(dataset))])
    Xi = torch.stack(Xi).to(device)
    Xv = torch.stack(Xv).to(device)
    with torch.no_grad():
        logits = model(Xi, Xv)
        rewards_sigmoid = torch.sigmoid(logits).squeeze().cpu().tolist()
        rewards = logits.squeeze().cpu().tolist()
    df["predicted_reward_sigmoid"] = rewards_sigmoid
    df["predicted_reward"] = rewards
    df.to_parquet(output_path, index=False)
    print(f"Saved predicted rewards to {output_path}")

def load_deepfm_model(model_path, feature_sizes=[19, 110, 2, 7, 1, 1, 1, 1, 1], embedding_size=8, shape=[800, 800, 800]):
    """Load a DeepFM model from a checkpoint file."""
    model = DeepFM(
        feature_sizes=feature_sizes,
        embedding_size=embedding_size,
        hidden_dims=shape
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model