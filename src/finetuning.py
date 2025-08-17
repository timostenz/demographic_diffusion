import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import GradScaler
scaler = GradScaler("cuda")
from diffusers import StableDiffusionPipeline
#from diffusers.models.attention_processor import LoRAAttnProcessor
#from diffusers.models.attention import Attention
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig, PeftModel
#from peft.tuners.lora import LoraLayer
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import UNet2DConditionModel, DDIMScheduler
from importlib import resources

from aesthetic_inference import *
from text_embedding_generator import *
from reward_function import *
from ddim_with_logprob import *


def load_pipeline(device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)

    pipe.enable_attention_slicing()
    return pipe


def generate_images(prompts, pipe):
    images = []
    for prompt in tqdm(prompts, desc="Generating images"):
        with torch.no_grad():
            image = pipe(prompt).images[0]
            images.append(image)
    return images


def compute_embeddings(images, prompts, device="cuda"):
    # Load image aesthetic model
    clip_model, preprocess = load_clip_model("ViT-L-14", device)
    aesthetic_model = get_aesthetic_model("vit_l_14", device)

    # Load text embedding model
    #tokenizer, text_model = load_clip_text_encoder(model_name="ViT-L-14", device=device)

    image_embeddings = []
    #text_embeddings = []
    aesthetic_scores = []

    for img, _ in tqdm(zip(images, prompts), total=len(prompts), desc="Computing embeddings"):
        #img_path = "/tmp/tmp_image.png"
        #img.save(img_path)
        
        # Compute image embeddings
        image_embed = get_clip_embedding_from_image(img, clip_model, preprocess, device)
        score = aesthetic_model(image_embed).item() if image_embed is not None else None

        # Compute text embeddings
        #text_embed = get_clip_text_embeddings([text], tokenizer, text_model, device=device)[0]

        image_embeddings.append(image_embed.squeeze(0).cpu().numpy().tolist())
        #text_embeddings.append(text_embed)
        aesthetic_scores.append(score)

    return image_embeddings, aesthetic_scores#, text_embeddings


def generate_images_embeddings(prompt_file, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompts and original metadata
    df = pd.read_parquet(prompt_file) if prompt_file.endswith(".parquet") else pd.read_csv(prompt_file)
    #df = df.head(20)
    prompts = df["prompt"].tolist()

    # Generate images
    pipe = load_pipeline(device)
    images = generate_images(prompts, pipe)

    # Compute embeddings
    image_embeds, scores = compute_embeddings(images, prompts, device)

    # Add results as new columns
    df["clip_embedding"] = image_embeds
    #df["clip_text_embedding"] = text_embeds
    df["aesthetic_score"] = scores

    #scaler = MinMaxScaler()
    #df[['aesthetic_score']] = scaler.fit_transform(
    #    df[['aesthetic_score']]
    #)

    # Apply fixed min-max scaling
    min_val = 0.21
    max_val = 8.37
    df["aesthetic_score"] = (df["aesthetic_score"] - min_val) / (max_val - min_val)

    # Save full enriched DataFrame
    df.to_parquet(output_file, index=False)
    print(f"Baseline data saved to {output_file}")

def compute_deepfm_reward(data_path, model_path, output_path):
    
    df = pd.read_parquet(data_path)
    #embedding_lookup_df = pd.read_parquet(embedding_lookup_path)

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


##################################################
######## RL FINE TUNING
##################################################

def load_pipeline_and_models(device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", safety_checker=None
    )
    pipe.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    pipe.to(device)
    pipe.enable_attention_slicing()

    # Apply LoRA via PEFT
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    # Sanity check: Print trainable parameter ratio
    #trainable = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    #total = sum(p.numel() for p in pipe.unet.parameters())
    #print(f"[LoRA Check] Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    # Print LoRA parameter names
    #print("[LoRA Check] Trainable parameter names:")
    #for name, param in pipe.unet.named_parameters():
    #    if param.requires_grad:
    #        print(" -", name)

    # Verify LoRA modules are injected
    #print("[LoRA Check] Injected LoRA modules:")
    #for name, module in pipe.unet.named_modules():
    #    if isinstance(module, LoraLayer):
    #        print(" ✅", name)

    return pipe

def evaluate_on_reward_prompts(pipe, df_queries, reward_model_fn, device="cuda"):

    clip_model, preprocess = load_clip_model("ViT-L-14", device)
    aesthetic_model = get_aesthetic_model("vit_l_14", device)

    def scale_aesthetic_score(aesthetic_score, min_val=0.21, max_val=8.37):
        return (aesthetic_score - min_val) / (max_val - min_val)

    rewards = []
    images = []
    image_embeds = []
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc="Evaluating reward prompts", disable=True):
        prompt = row["prompt"]
        image = pipe(prompt).images[0]

        #image_embed = get_clip_embedding_from_image(image, clip_model, preprocess, device)
        #aesthetic_score = aesthetic_model(image_embed).item()

        # normalize reward as done in synthetic dataset
        #aesthetic_score = scale_aesthetic_score(aesthetic_score)

        #reward = reward_model_fn(aesthetic_score, row)
        
        #just aesthetic score
        reward = reward_model_fn(image)

        rewards.append(reward)
        #images.append(image)
        #image_embeds.append(image_embed)
    rewards_tensor = torch.tensor(rewards)
    avg_reward = rewards_tensor.mean().item()
    std_reward = rewards_tensor.std().item()
    print(f"[Validation] Avg Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    return avg_reward, std_reward, rewards_tensor#, images, image_embeds

def rl_finetune_loop(
    pipe, df_prompts, df_queries=None, reward_model_fn=None,
    num_epochs=1, batch_size=256, device="cuda", rl_method="rwr", use_kl_reg=False
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    pipe.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    pipe.unet.train()
    pipe.scheduler.set_timesteps(50)

    # datasets for evaluation
    # df queries 500 random samples for evaluation
    # df prompts 64 random prompts; mens clothing category
    # df to evaluate unseen prompts from same category
    df_male = pd.read_parquet("df_evaluation_male.parquet")
    # df to evaluate women clothing
    df_female = pd.read_parquet("df_evaluation_female.parquet")
    # df home and kitchen (check generalization)
    df_kitchen = pd.read_parquet("df_evaluation_kitchen.parquet")

    reward_log = []
    reward_log_epoch1 = []
    reward_log_train = []
    reward_log_male = []
    reward_log_female = []
    reward_log_kitchen = []
    evaluation_thresholds = {0, 2_560, 5_120, 7_680, 10_240, 12_800, 15_360, 17_920, 20_480, 23_040, 25_600}
    eval_checkpoint_set = set()
    prompts_seen = 0
    loss_log = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        #df_epoch = df_prompts.sample(frac=1).reset_index(drop=True)
        num_batches_per_epoch = 100

        # Evaluation at 0 prompts (before any fine-tuning)
        if epoch == 0 and 0 in evaluation_thresholds and 0 not in eval_checkpoint_set:
            if df_queries is not None and reward_model_fn is not None:
                avg_reward, std_reward, _ = evaluate_on_reward_prompts(pipe, df_queries, reward_model_fn, device)
                reward_log_epoch1.append({
                    "epoch": "step_0",
                    "val_avg_reward": avg_reward,
                    "val_std_reward": std_reward
                })
                avg_reward_train, std_reward_train, _ = evaluate_on_reward_prompts(pipe, df_prompts, reward_model_fn, device)
                reward_log_train.append({
                    "epoch": "step_0",
                    "val_avg_reward": avg_reward_train,
                    "val_std_reward": std_reward_train
                })
                avg_reward_male, std_reward_male, _ = evaluate_on_reward_prompts(pipe, df_male, reward_model_fn, device)
                reward_log_male.append({
                    "epoch": "step_0",
                    "val_avg_reward": avg_reward_male,
                    "val_std_reward": std_reward_male
                })
                avg_reward_female, std_reward_female, _ = evaluate_on_reward_prompts(pipe, df_female, reward_model_fn, device)
                reward_log_female.append({
                    "epoch": "step_0",
                    "val_avg_reward": avg_reward_female,
                    "val_std_reward": std_reward_female
                })
                avg_reward_kitchen, std_reward_kitchen, _ = evaluate_on_reward_prompts(pipe, df_kitchen, reward_model_fn, device)
                reward_log_kitchen.append({
                    "epoch": "step_0",
                    "val_avg_reward": avg_reward_kitchen,
                    "val_std_reward": std_reward_kitchen
                })
                eval_checkpoint_set.add(0)
                print(f"Step 0: Test (500 random prompts) avg_reward={avg_reward:.4f}, std={std_reward:.4f}")
                print(f"Step 0: Train (Male Clothing) avg_reward={avg_reward_train:.4f}, std={std_reward_train:.4f}")
                print(f"Step 0: Test (Male Clothing) avg_reward={avg_reward_male:.4f}, std={std_reward_male:.4f}")
                print(f"Step 0: Test (Female Clothing) avg_reward={avg_reward_female:.4f}, std={std_reward_female:.4f}")
                print(f"Step 0: Test (Kitchen) avg_reward={avg_reward_kitchen:.4f}, std={std_reward_kitchen:.4f}")

        for _ in tqdm(range(num_batches_per_epoch), desc="Fine-tuning"):
            #batch_df = df_prompts.sample(n=batch_size)
            batch_df = pd.concat([df_prompts] * 4, ignore_index=True)
            prompts = batch_df["prompt"].tolist()

            with torch.no_grad():
                # Generate images with current model
                #images = generate_images(prompts, pipe)

                # Compute embeddings & aesthetic scores
                #image_embeds, scores = compute_embeddings(images, prompts, device)

                # Apply fixed min-max scaling
                #min_val = 0.21
                #max_val = 8.37
                #scores = torch.tensor(scores, dtype=torch.float32)
                #scores = (scores - min_val) / (max_val - min_val + 1e-6)
                #scores = scores.tolist()

                # Compute structured features for reward model
                #structured_feats = extract_structured_features(batch, device)  # <- you must define this!

                # Compute live rewards
                _, _, rewards = evaluate_on_reward_prompts(pipe, batch_df, reward_model_fn, device)
                # Normalize rewards in PyTorch
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                rewards = rewards.tolist()
            prompts_seen += len(prompts)

            text_inputs = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs)[0]
            #latents = torch.randn((len(prompts), pipe.unet.config.in_channels, 64, 64), device=device)
            #timestep = pipe.scheduler.timesteps[25]
            #latents_input = latents.clone().detach().requires_grad_(True)

            #noise_pred = pipe.unet(
            #    sample=latents_input,
            #    timestep=timestep,
            #    encoder_hidden_states=encoder_hidden_states
            #).sample

            unet_base = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="unet"
            ).to(device).eval()
            unet_base.requires_grad_(False)

            #rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            #if rl_method == "rwr":
            #
            #    if use_kl_reg:
            #        with torch.no_grad():
            #            noise_pred_old = unet_base(sample=latents_input, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
            #        kl = F.mse_loss(noise_pred, noise_pred_old, reduction="none").view(len(prompts), -1).mean(dim=1).mean()
            #        kl_coeff = 0.01
            #    else:
            #        kl = 0.0
            #        kl_coeff = 0.0
            #
            #    beta = 0.5  # or make configurable
            #    sharpened = torch.exp(beta * rewards_tensor)
            #    loss_main = (sharpened * (noise_pred ** 2)).mean()
            #    #print("KL penalty:", kl_coeff * kl)
            #    loss = loss_main + kl_coeff * kl
            #
            #    optimizer.zero_grad()
            #    loss.backward()
            #    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            #    optimizer.step()

            #elif rl_method == "rwrsparse":
            #
            #    if use_kl_reg:
            #        with torch.no_grad():
            #            noise_pred_old = unet_base(sample=latents_input, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
            #        kl = F.mse_loss(noise_pred, noise_pred_old, reduction="none").view(len(prompts), -1).mean(dim=1).mean()
            #        kl_coeff = 0.01
            #    else:
            #        kl = 0.0
            #        kl_coeff = 0.0
            #
            #    rewards_flat = torch.tensor(rewards, dtype=torch.float32, device=device)
            #    threshold = torch.quantile(rewards_flat, 0.9)
            #    mask = rewards_flat >= threshold
            #    loss_per_sample = (rewards_tensor * (noise_pred ** 2)).view(len(prompts), -1).mean(dim=1)
            #    loss_main = loss_per_sample[mask].mean()
            #    #print("KL penalty:", kl_coeff * kl)
            #    loss = loss_main + kl_coeff * kl
            #
            #    optimizer.zero_grad()
            #    loss.backward()
            #    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            #    optimizer.step()

            if rl_method == "ddposf":
                unet_frozen = UNet2DConditionModel.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", subfolder="unet"
                ).to(device).eval()
                unet_frozen.requires_grad_(False)
                optimizer.zero_grad()
                loss = collect_ddpo_trajectory(
                    pipe, prompts, encoder_hidden_states,
                    pipe.scheduler, device, rewards, unet_frozen,
                    optimizer, use_kl_reg=use_kl_reg, use_true_logprob=True,
                    T=50, T_sample=50, grad_accum_steps=256
                )
                loss_log.append(loss)
                #optimizer.step()

            elif rl_method == "ddpois":
                #now calles before the if statements
                # base sd 1.4 for Kl reg
                #unet_base = UNet2DConditionModel.from_pretrained(
                #    "CompVis/stable-diffusion-v1-4", subfolder="unet"
                #).to(device).eval()
                #unet_base.requires_grad_(False)

                # last step model for gradient
                unet_frozen = copy.deepcopy(pipe.unet)
                unet_frozen = unet_frozen.merge_and_unload()  # merge LoRA into base weights
                unet_frozen.eval().to(device)
                #log_new, log_old, rewards_tensor = collect_ddpois_trajectory(
                #    pipe, prompts, encoder_hidden_states, pipe.scheduler, device, reward_model_fn, unet_old=unet_frozen
                #)
                #loss = ddpois_loss(log_new, log_old, rewards_tensor)
                optimizer.zero_grad()
                loss = collect_ddpois_trajectory(
                    pipe, prompts, encoder_hidden_states, pipe.scheduler, device, rewards,
                    unet_old=unet_frozen, unet_base=unet_base, use_kl_reg=use_kl_reg
                )
                optimizer.step()

            else:
                raise ValueError(f"Unknown rl_method: {rl_method}")
            
            # Mid-epoch evaluation checkpoints (only in first epoch)
            if epoch == 0 and prompts_seen in evaluation_thresholds and prompts_seen not in eval_checkpoint_set:
                if df_queries is not None and reward_model_fn is not None:
                    avg_reward, std_reward, _ = evaluate_on_reward_prompts(pipe, df_queries, reward_model_fn, device)
                    reward_log_epoch1.append({
                        "epoch": f"step_{prompts_seen}",
                        "val_avg_reward": avg_reward,
                        "val_std_reward": std_reward
                    })
                    avg_reward_train, std_reward_train, _ = evaluate_on_reward_prompts(pipe, df_prompts, reward_model_fn, device)
                    reward_log_train.append({
                        "epoch": f"step_{prompts_seen}",
                        "val_avg_reward": avg_reward_train,
                        "val_std_reward": std_reward_train
                    })
                    avg_reward_male, std_reward_male, _ = evaluate_on_reward_prompts(pipe, df_male, reward_model_fn, device)
                    reward_log_male.append({
                        "epoch": f"step_{prompts_seen}",
                        "val_avg_reward": avg_reward_male,
                        "val_std_reward": std_reward_male
                    })
                    avg_reward_female, std_reward_female, _ = evaluate_on_reward_prompts(pipe, df_female, reward_model_fn, device)
                    reward_log_female.append({
                        "epoch": f"step_{prompts_seen}",
                        "val_avg_reward": avg_reward_female,
                        "val_std_reward": std_reward_female
                    })
                    avg_reward_kitchen, std_reward_kitchen, _ = evaluate_on_reward_prompts(pipe, df_kitchen, reward_model_fn, device)
                    reward_log_kitchen.append({
                        "epoch": f"step_{prompts_seen}",
                        "val_avg_reward": avg_reward_kitchen,
                        "val_std_reward": std_reward_kitchen
                    })
                    eval_checkpoint_set.add(prompts_seen)
                    print(f"Step {prompts_seen}: Test (500 random prompts) avg_reward={avg_reward:.4f}, std={std_reward:.4f}")
                    print(f"Step {prompts_seen}: Train (Male Clothing) avg_reward={avg_reward_train:.4f}, std={std_reward_train:.4f}")
                    print(f"Step {prompts_seen}: Test (Male Clothing) avg_reward={avg_reward_male:.4f}, std={std_reward_male:.4f}")
                    print(f"Step {prompts_seen}: Test (Female Clothing) avg_reward={avg_reward_female:.4f}, std={std_reward_female:.4f}")
                    print(f"Step {prompts_seen}: Test (Kitchen) avg_reward={avg_reward_kitchen:.4f}, std={std_reward_kitchen:.4f}")

        if df_queries is not None and reward_model_fn is not None:
            avg_reward, std_reward, _ = evaluate_on_reward_prompts(pipe, df_queries, reward_model_fn, device)
            reward_log.append({
                "epoch": epoch + 1,
                "val_avg_reward": avg_reward,
                "val_std_reward": std_reward
            })

    #pipe.save_pretrained("rl_finetuned_sd")
    #pipe.unet.save_attn_procs("lora_unet_finetuned")

    folder_name = f"finetuned_model_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot"
    os.makedirs(folder_name, exist_ok=True)

    pipe.unet = pipe.unet.merge_and_unload()
    pipe.unet.save_pretrained(f"{folder_name}/unet_full_merged")
    pipe.text_encoder.save_pretrained(f"{folder_name}/text_encoder_finetuned")
    pipe.scheduler.save_pretrained(f"{folder_name}/scheduler_finetuned")

    print("Saved fine-tuned model.")

    filename_epoch1 = f"reward_tracking_test500_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    pd.DataFrame(reward_log_epoch1).to_csv(filename_epoch1, index=False)
    print(f"Saved reward tracking (test 500) log to {filename_epoch1}")

    filename_train = f"reward_tracking_train_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    pd.DataFrame(reward_log_train).to_csv(filename_train, index=False)
    print(f"Saved reward tracking (train) log to {filename_train}")

    filename_male = f"reward_tracking_male_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    pd.DataFrame(reward_log_male).to_csv(filename_male, index=False)
    print(f"Saved reward tracking (male) log to {filename_male}")

    filename_female = f"reward_tracking_female_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    pd.DataFrame(reward_log_female).to_csv(filename_female, index=False)
    print(f"Saved reward tracking (female) log to {filename_female}")

    filename_kitchen = f"reward_tracking_kitchen_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    pd.DataFrame(reward_log_kitchen).to_csv(filename_kitchen, index=False)
    print(f"Saved reward tracking (kitchen) log to {filename_kitchen}")

    #filename = f"reward_tracking_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    #pd.DataFrame(reward_log).to_csv(filename, index=False)
    #print(f"Saved reward tracking log to {filename}")

    #filename = f"loss_tracking_{rl_method}_kl_{str(use_kl_reg).lower()}_fewshot.csv"
    #pd.DataFrame(loss_log).to_csv(filename, index=False)
    #print(f"Saved loss tracking log to {filename}")

def make_deepfm_reward_function(model_path, device="cuda"):
    model = load_deepfm_model(model_path)
    model.eval().to(device)

    def reward_fn(aesthetic_score, row):
        with torch.no_grad():
            Xi = torch.tensor([[
                row["main_category"],
                row["sub_category"],
                row["gender"],
                row["age_group"],
                0, 0, 0, 0, 0
            ]], dtype=torch.long).to(device)

            Xv = torch.tensor([[
                1.0, 1.0, 1.0, 1.0,
                aesthetic_score,
                row["ratings"],
                row["discount_percentage"],
                row["discount_price_log"],
                row["actual_price_log"]
            ]], dtype=torch.float32).to(device)

            reward = model(Xi, Xv).item()
            return reward

    return reward_fn

def collect_ddpo_trajectory(pipe, prompts, text_embeddings, scheduler, device, rewards, unet_frozen, optimizer,
                            grad_accum_steps=4, T=5, use_kl_reg=False, kl_coeff=0.01,
                            T_sample=5, use_true_logprob=False, eta=1.0):
    batch_size = len(prompts)
    assert batch_size % grad_accum_steps == 0, "Batch size must be divisible by grad_accum_steps"
    mini_batch_size = batch_size // grad_accum_steps

    total_loss = 0.0
    for i in range(grad_accum_steps):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        if end - start != mini_batch_size:
            continue

        sub_prompts = prompts[start:end]
        sub_embeddings = text_embeddings[start:end]
        sub_rewards = rewards[start:end]

        latent_height = pipe.unet.config.sample_size
        latent_width = pipe.unet.config.sample_size
        latent_shape = (mini_batch_size, pipe.unet.config.in_channels, latent_height, latent_width)
        latents = torch.randn(latent_shape, device=device) * pipe.scheduler.init_noise_sigma

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = latents
            log_probs = []
            kl_records = [] if use_kl_reg else None
            noise_target = torch.zeros_like(x)

            for t in scheduler.timesteps[:T]:
                step_input = x.detach()

                noise_pred = pipe.unet(
                    sample=step_input,
                    timestep=t,
                    encoder_hidden_states=sub_embeddings
                ).sample

                if use_true_logprob:
                    x_prev, logp = ddim_step_with_logprob(
                        scheduler,
                        model_output=noise_pred,
                        timestep=t,
                        sample=step_input,
                        eta=eta,
                    )
                else:
                    logp = -F.mse_loss(noise_pred, noise_target, reduction="none").view(mini_batch_size, -1).mean(dim=1)
                    with torch.no_grad():
                        x_prev = scheduler.step(noise_pred, t, step_input).prev_sample

                log_probs.append(logp)
                x = x_prev

                if use_kl_reg:
                    with torch.no_grad():
                        noise_pred_old = unet_frozen(sample=step_input, timestep=t, encoder_hidden_states=sub_embeddings).sample

                    if use_true_logprob:
                        _, logp_old = ddim_step_with_logprob(
                            scheduler,
                            model_output=noise_pred_old,
                            timestep=t,
                            sample=step_input,
                            eta=eta,
                            prev_sample=x_prev
                        )
                        kl = (logp_old - logp).detach()
                    else:
                        kl = F.mse_loss(noise_pred, noise_pred_old, reduction="none").view(mini_batch_size, -1).mean(dim=1)

                    kl_records.append(kl)

            log_probs = torch.stack(log_probs, dim=1)
            rewards_tensor = torch.tensor(sub_rewards, dtype=torch.float32, device=device)
            base_loss = ddposf_loss(log_probs, rewards_tensor, T_sample=T_sample)

            if use_kl_reg:
                kl_matrix = torch.stack(kl_records, dim=1)
                ts = torch.randint(0, kl_matrix.shape[1], (mini_batch_size, T_sample), device=device)
                selected_kl = kl_matrix.gather(1, ts).mean(dim=1).mean()
                total_step_loss = (base_loss + kl_coeff * selected_kl) / grad_accum_steps
            else:
                total_step_loss = base_loss / grad_accum_steps

        #retain = i < grad_accum_steps - 1
        scaler.scale(total_step_loss).backward(retain_graph=True)
        total_loss += total_step_loss.item()

    scaler.step(optimizer)
    scaler.update()

    print("total loss: ", total_loss)
    return total_loss

def ddposf_loss(log_probs, rewards, T_sample=10):
    B, T = log_probs.shape
    ts = torch.randint(0, T, (B, T_sample), device=log_probs.device)
    selected_logp = log_probs.gather(1, ts)  # shape [B, T_sample]
    return - (selected_logp.mean(dim=1) * rewards).mean()

def collect_ddpois_trajectory(
    pipe, prompts, text_embeddings, scheduler, device,
    rewards, unet_old, optimizer, unet_base, grad_accum_steps=4, T=5, use_kl_reg=False, kl_coeff=0.01,
    T_sample=5, use_true_logprob=False, eta=1.0
):
    batch_size = len(prompts)
    assert batch_size % grad_accum_steps == 0, "Batch size must be divisible by grad_accum_steps"
    mini_batch_size = batch_size // grad_accum_steps

    total_loss = 0.0

    for i in range(grad_accum_steps):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        if end - start != mini_batch_size:
            continue

        sub_prompts = prompts[start:end]
        sub_embeddings = text_embeddings[start:end]
        sub_rewards = rewards[start:end]

        latent_height = pipe.unet.config.sample_size
        latent_width = pipe.unet.config.sample_size
        latent_shape = (mini_batch_size, pipe.unet.config.in_channels, latent_height, latent_width)
        latents = torch.randn(latent_shape, device=device) * pipe.scheduler.init_noise_sigma

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = latents
            log_new, log_old = [], []
            kl_records = [] if use_kl_reg else None

            noise_target = torch.zeros_like(x)

            for t in scheduler.timesteps[:T]:
                step_input = x.detach()

                noise_pred = pipe.unet(
                    sample=step_input,
                    timestep=t,
                    encoder_hidden_states=sub_embeddings
                ).sample

                with torch.no_grad():
                    noise_pred_old = unet_old(
                        sample=step_input,
                        timestep=t,
                        encoder_hidden_states=sub_embeddings
                    ).sample

                if use_true_logprob:
                    print("continue here!!!!!")

            logp_new = -F.mse_loss(noise_pred, noise_target, reduction="none").view(mini_batch_size, -1).mean(dim=1)
            logp_old = -F.mse_loss(noise_pred_old, noise_target, reduction="none").view(mini_batch_size, -1).mean(dim=1)

            log_new.append(logp_new)
            log_old.append(logp_old)

            if use_kl_reg:
                with torch.no_grad():
                    noise_pred_old = unet_base(sample=step_input, timestep=t, encoder_hidden_states=sub_embeddings).sample
                kl = F.mse_loss(noise_pred, noise_pred_old, reduction="none").view(mini_batch_size, -1).mean(dim=1)
                kl_records.append(kl)
                #print(f"[KL Step] t={t}: KL mean = {kl.mean().item():.8f}")

            with torch.no_grad():
                x = scheduler.step(noise_pred, t, step_input).prev_sample

        log_new = torch.stack(log_new, dim=1)
        log_old = torch.stack(log_old, dim=1)
        rewards_tensor = torch.tensor(sub_rewards, dtype=torch.float32, device=device)

        base_loss = ddpois_loss(log_new, log_old, rewards_tensor)

        if use_kl_reg:
            kl_matrix = torch.stack(kl_records, dim=1)  # [B, T]
            ts = torch.randint(0, kl_matrix.shape[1], (mini_batch_size, T), device=device)
            selected_kl = kl_matrix.gather(1, ts).mean(dim=1).mean()  # mean over sample and batch
            #print("KL penalty:", selected_kl)
            total_step_loss = (base_loss + kl_coeff * selected_kl) / grad_accum_steps
        else:
            total_step_loss = base_loss / grad_accum_steps

        retain = i < grad_accum_steps - 1
        total_step_loss.backward(retain_graph=retain)
        total_loss += total_step_loss.item()

    return total_loss

def ddpois_loss(log_probs_new, log_probs_old, rewards, clip_weight=5.0):
    rewards = rewards.unsqueeze(1)  # shape [B, 1]
    ratio = (log_probs_new - log_probs_old).detach().exp().clamp(max=clip_weight)  # shape: [B, T]
    loss = -((ratio * rewards) * log_probs_new).mean()  # shape: [B, T]
    return loss

def make_aesthetic_score_function():
    #from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

#ASSETS_PATH = resources.files("ddpo_pytorch.assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load("sac+logos+ava1-l14-linearMSE.pth")
        #    ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        #)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)