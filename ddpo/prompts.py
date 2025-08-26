from importlib import resources
import os
import functools
import random
import inflect
import pandas as pd

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def structured_prompts_maletrain():
    structured_data = pd.read_parquet("assets/df_finetuning_male45.parquet")
    row = structured_data.sample(1).iloc[0]
    prompt = row["prompt"]
    metadata = row.to_dict()  # include all relevant fields
    return prompt, metadata

def structured_prompts_maleeval():
    structured_data = pd.read_parquet("assets/df_evaluation_male45.parquet")
    row = structured_data.sample(1).iloc[0]
    prompt = row["prompt"]
    metadata = row.to_dict()  # include all relevant fields
    return prompt, metadata

def structured_prompts_femaleeval():
    structured_data = pd.read_parquet("assets/df_evaluation_female45.parquet")
    row = structured_data.sample(1).iloc[0]
    prompt = row["prompt"]
    metadata = row.to_dict()  # include all relevant fields
    return prompt, metadata

def structured_prompts_kitcheneval():
    structured_data = pd.read_parquet("assets/df_evaluation_kitchen45.parquet")
    row = structured_data.sample(1).iloc[0]
    prompt = row["prompt"]
    metadata = row.to_dict()  # include all relevant fields
    return prompt, metadata

def male_train():
    return from_file("prompts_male_train45.txt")

def male_evaluate():
    return from_file("prompts_male_evaluation45.txt")

def female_evaluate():
    return from_file("prompts_female_evaluation45.txt")

def kitchen_evaluate():
    return from_file("prompts_kitchen_evaluation45.txt")