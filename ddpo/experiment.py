import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def deep_fm():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 200
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4

    # prompting
    config.prompt_fn = "structured_prompts_maletrain"
    config.prompt_fn2 = "structured_prompts_maleeval"
    config.prompt_fn3 = "structured_prompts_femaleeval"
    config.prompt_fn4 = "structured_prompts_kitcheneval"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    # rewards
    config.reward_fn = "deepfm_score"

    return config

def get_config(name):
    return globals()[name]()
