
# Demographic Diffusion

This repository demonstrates a proof of concept: a diffusion model can be successfully fine-tuned using reinforcement learning and functions common in recommender systems. The workflow involves creating a synthetic dataset, learning a reward function, and then fine-tuning the diffusion model. All code related to data generation and reward training is located in the `src` directory, while diffusion model fine-tuning is handled in the `ddpo` directory.

This code was developed for a research study, which will be linked here once it becomes publicly available.

## The `src` Directory

The `src` part of the repository provides a modular pipeline for generating synthetic data, running aesthetic inference, training reward models, and learning *demographic regularizers* to guide diffusion model fine-tuning.

## Workflow Overview


1. **Data Creation**
	 - Generate a synthetic dataset from your raw product data. (We used this dataset: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/)
	 - Command:
		 ```bash
		 python src/main.py --task create_data
		 ```

2. **Aesthetic Inference**
	 - Compute aesthetic scores for images using CLIP and an aesthetic model.
	 - Command:
		 ```bash
		 python src/main.py --task aesthetic_inference
		 ```

3. **Expand Data**
	 - Expand and preprocess the data for training (e.g., one row per customer).
	 - Command:
		 ```bash
		 python src/main.py --task expand_dataset
		 ```

4. **Train Reward Function**
	 - Train a reward model (DeepFM) to predict user engagement or preferences.
	 - Command:
		 ```bash
		 python src/main.py --task train_reward_function
		 ```

5. **Evaluate Reward Model**
	 - Evaluate the trained reward model on a test set.
	 - Command:
		 ```bash
		 python src/main.py --task evaluate
		 ```

6. **Train Gender Probe**
	 - Train a probe to predict gender from embeddings (for demographic regularization).
	 - Command:
		 ```bash
		 python src/main.py --task train_clip_gender_probe
		 ```

7. **Train Age Probe**
	 - Train a probe to predict age group from embeddings (for demographic regularization, using aggregated age groups).
	 - Command:
		 ```bash
		 python src/main.py --task train_clip_age_probe_aggregated
		 ```





---

## The `ddpo` Directory (Diffusion Model Fine-Tuning)

This part of the repository is used for fine-tuning diffusion models using reinforcement learning with custom reward functions and prompts.

### Usage

To start fine-tuning Stable Diffusion v1.4 on all available GPUs using the DGX config:

```bash
accelerate launch scripts/train.py --config config/dgx.py:aesthetic
```

This will immediately start the fine-tuning process. The default config is set up for multi-GPU training.

### Important Hyperparameters

- All hyperparameters are defined in the config files (see `base.py` and `dgx.py`).
- **prompt_fn** and **reward_fn**: Define the prompts and reward functions for training. See `ddpo_pytorch/prompts.py` and `ddpo_pytorch/rewards.py` for available options. This is also where you can find a new recommender systems reward function.


This part is an extension of the [ddpo-pytorch repository](https://github.com/kvablack/ddpo-pytorch). For more details, see the comments in the config files and the original repository.

---

For questions or comments, please contact: timo.stenz@tuebingen.mpg.de