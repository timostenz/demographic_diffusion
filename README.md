
# Demographic Diffusion

This repository demonstrates a proof of concept: a diffusion model can be successfully fine-tuned using reinforcement learning and functions common in recommender systems. The workflow involves creating a synthetic dataset, learning a reward function, and then fine-tuning the diffusion model. All code related to data generation and reward training is located in the `src` directory, while diffusion model fine-tuning is handled in the `ddpo` directory.

This code was developed for a research study, which will be linked here once it becomes publicly available.

## Workflow Overview

To replicate our results, please follow these steps.


1. Create a synthetic dataset from raw product data (using the [Amazon-Products.csv](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/) dataset).
    ```bash
    python src/main.py --task create_data
    ```

2. Compute aesthetic scores for images using CLIP embeddings.
    ```bash
    python src/main.py --task aesthetic_inference
    ```

3. Bring the data to the individual level.
    ```bash
    python src/main.py --task expand_dataset
    ```

4. Train a reward model to predict user engagement.
    ```bash
    python src/main.py --task train_reward_function
    ```

5. Evaluate the trained reward model on a test set.
    ```bash
    python src/main.py --task evaluate
    ```

6. Train a classifier to predict gender from embeddings (for demographic regularization).
    ```bash
    python src/main.py --task train_clip_gender_probe
    ```

7. Train a classifier to predict age group from embeddings.
    ```bash
    python src/main.py --task train_clip_age_probe_aggregated
    ```

8. Start fine-tuning Stable Diffusion v1.4 on all available GPUs using the experiment config. The default config is set up for multi-GPU training.

    ```bash
    accelerate launch scripts/train.py --config config/experiment.py:deep_fm
    ```



## The `ddpo` Directory

This part of the repository is used for fine-tuning diffusion models using reinforcement learning with custom reward functions and prompts. It uses the learned reward function and demographic probes from the `src` part. The code is an extension of the [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) repository. For more details, see the comments in the config files and the original repository.

---

For questions or comments, please contact: timo.stenz@tuebingen.mpg.de