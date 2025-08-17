import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import open_clip
from tqdm import tqdm

# Loaders
def load_bert(device="cpu"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    return tokenizer, model

def load_clip_text_encoder(model_name="ViT-L-14", device="cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()
    return tokenizer, model

# Embedding extractors
def get_bert_embeddings(text_batch, tokenizer, model, device="cpu"):
    inputs = tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embeddings.cpu().numpy().tolist()
    
def get_clip_text_embeddings(text_batch, tokenizer, model, device="cpu"):
    tokens = tokenizer(text_batch).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens).float()
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy().tolist()

# Core function
def process_text_column(
    df, column_name="name", model_type="clip", device=None, batch_size=32,
    save_interval=20000, output_path=None, return_df=False
):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose model
    if model_type == "bert":
        tokenizer, model = load_bert(device)
        get_embeddings = lambda batch: get_bert_embeddings(batch, tokenizer, model, device)
        column_name_out = "bert_embedding"
    elif model_type == "clip":
        tokenizer, model = load_clip_text_encoder(device=device)
        get_embeddings = lambda batch: get_clip_text_embeddings(batch, tokenizer, model, device)
        column_name_out = "clip_text_embedding"
    else:
        raise ValueError("Invalid model_type. Choose from 'bert' or 'clip'.")

    # Resume logic
    if column_name_out in df.columns:
        processed_rows = df[column_name_out].notna().sum()
    else:
        df[column_name_out] = None
        processed_rows = 0

    texts = df[column_name].tolist()

    for i in tqdm(range(processed_rows, len(texts), batch_size), desc=f"Processing {model_type.upper()} Text Embeddings"):
        batch_texts = texts[i: i + batch_size]
        batch_embeddings = get_embeddings(batch_texts)
        df[column_name_out].iloc[i:i + len(batch_embeddings)] = batch_embeddings

        if output_path and i % save_interval == 0:
            df.to_parquet(output_path, index=False)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df if return_df else None

# Convenience wrapper
def process_parquet_text(input_path, column_name="name", output_path=None, model_type="clip", **kwargs):
    df = pd.read_parquet(input_path)
    return process_text_column(df, column_name=column_name, output_path=output_path, model_type=model_type, **kwargs)