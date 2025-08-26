from PIL import Image
import io
import numpy as np
import torch

def aesthetic_score():
    from aesthetic_scorer import AestheticScorer

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

def deepfm_score(device=None):
    from src.reward_function import DeepFM
    from aesthetic_scorer import AestheticScorer

    def load_deepfm_model(model_path, feature_sizes=[19, 110, 2, 7, 1, 1, 1, 1, 1], embedding_size=8, shape=[800, 800, 800], device=device):

        model = DeepFM(
            feature_sizes=feature_sizes,
            embedding_size=embedding_size,
            hidden_dims=shape
        )

        device = device
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        return model
    
    def make_deepfm_reward_function(model_path, device=device):
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

    deepfm = make_deepfm_reward_function("assets/deepfm2000fusion_new4_bs2048_lr1e-3_drop0.5_shape_embeddingdim8_epochs6_count1.pt")
    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    
        scores = scorer(images)

        min_val = 0.21
        max_val = 8.37
        scores = (scores - min_val) / (max_val - min_val + 1e-6)

        deepfm_scores = []

        for score, row in zip(scores, metadata):
            reward = deepfm(score.item(), row)
            deepfm_scores.append(reward)

        deepfm_scores = [deepfm(score, row) for score, row in zip(scores, metadata)]
        return deepfm_scores, {}

    return _fn