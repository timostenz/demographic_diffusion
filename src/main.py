import argparse
import pandas as pd
import os
import json
from aesthetic_inference import *
from create_data import *
from change_dataformat import *
from text_embedding_generator import *
from reward_function import *
from finetuning import *

def main():
    parser = argparse.ArgumentParser(description="Run aesthetic inference or create synthetic data.")
    
    # Common argument to choose task
    parser.add_argument("--task", type=str, choices=["aesthetic_inference",
                                                     "clip_text_inference",
                                                     "create_data",
                                                     "expand_dataset",
                                                     "prepare_training",
                                                     "train_reward_function",
                                                     "evaluate",
                                                     "train_clip_gender_probe",
                                                     "train_clip_age_probe",
                                                     "train_clip_age_probe_aggregated",
                                                     "generate_images_embeddings", 
                                                     "get_image_rewards",
                                                     "rl_finetune"], required=True, help="Choose the task to run.")

    # Arguments for aesthetic inference & BERT inference
    parser.add_argument("--parquet_path", type=str, help="Path to the input parquet file (for aesthetic inference).")
    parser.add_argument("--column_name", type=str, help="Name of the column containing image URLs.")
    parser.add_argument("--output", type=str, default="output.parquet", help="Path to save the processed parquet file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images to process in a batch.")
    parser.add_argument("--save_interval", type=int, default=20000, help="How often to save progress.")
    parser.add_argument("--drop_invalid", action="store_true", help="Drop rows with invalid image links.")

    # Arguments for synthetic data creation
    parser.add_argument("--input_file", type=str, default="Amazon-Products.csv", help="Path to input data file for synthetic data generation.")
    parser.add_argument("--output_file", type=str, default="synthetic_data.parquet", help="Path to save generated data.")

    # Arguments to change data format (preprocess and expand data)
    parser.add_argument("--input_file_changeformat", type=str, help="Path to input parquet file.")
    parser.add_argument("--output_file_changeformat", type=str, help="Path to save processed parquet file.")
    parser.add_argument("--expansion_factor", type=int, default=1000, help="How many rows on the individual level are created. Minimum of 1000 is recommended.")
    parser.add_argument("--lookup_file", action="store_true", help="Create a lookup table for embeddings.")

    # arguments to prepare training and train reward function
    parser.add_argument("--train_data_path", type=str, help="Path to expanded dataset with features for DeepFM.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model_out", type=str, default="deepfm_model.pt", help="Path to save the trained DeepFM model.")
    parser.add_argument("--loss_out", type=str, default="loss_history.csv", help="Path to save the loss history CSV.")
    parser.add_argument("--embedding_lookup_path", type=str, default="embedding_dict.parquet", help="Path to the embedding lookup table for ID-based training.")
    parser.add_argument("--batch_size_training", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for training.")
    parser.add_argument("--embedding_size", type=int, default=10, help="FM Embedding size.")
    parser.add_argument("--shape", type=int, nargs='+', default=[800, 800, 800], help="Network shape.")
    parser.add_argument('--dem_probe_path', nargs='*', default=[])
    parser.add_argument('--lambda_dem', nargs='*', type=float, default=[])
    parser.add_argument('--dem_label_index', nargs='*', type=int, default=[])

    # arguments for evaluation
    parser.add_argument("--test_data_path", type=str, help="Path to the test dataset for evaluation.")

    # arguments for probe
    parser.add_argument("--embedding_type", type=str, choices=["image", "text"], default="image", help="Which CLIP embedding to use for probing: 'image' or 'text'")
    parser.add_argument("--onevsrest", action="store_true", help="Use one-vs-rest logistic regression")

    # arguments for prompt-only baseline
    parser.add_argument("--prompt_file", type=str, help="Path to file with prompts for baseline generation.")
    parser.add_argument("--output_baseline", type=str, help="Path to save baseline generation output.")
    parser.add_argument("--output_rewards", type=str, help="Path to save reward generation output.")

    #arguments for rl fine tuning
    parser.add_argument("--output_diffusion_path", type=str, default="rl_finetuned_sd", help="Where to save the fine-tuned model.")
    parser.add_argument("--rl_method", type=str, default="rwr", help="RL method used for fine-tuning.")
    parser.add_argument("--use_kl_reg", action="store_true", help="Use KL regularization during fine tuning.")

    args = parser.parse_args()

    if args.task == "aesthetic_inference":
        if not args.parquet_path or not args.column_name:
            raise ValueError("For aesthetic inference, --parquet_path and --column_name must be provided.")
        
        # Load the dataset
        df = pd.read_parquet(args.parquet_path)
        
        # Process the images in the specified column
        process_image_column(
            df,
            args.column_name,
            drop_invalid=args.drop_invalid,
            output_path=args.output,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
        
        print(f"Aesthetic inference completed. Results saved to {args.output}")

    elif args.task == "clip_text_inference":
        if not args.parquet_path or not args.column_name:
            raise ValueError("For CLIP text inference, --parquet_path and --column_name must be provided.")

        df = pd.read_parquet(args.parquet_path)

        process_text_column(
            df,
            column_name=args.column_name,
            output_path=args.output,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            return_df=False,
            model_type="clip"  # <--- forces CLIP usage
        )

        print(f"CLIP text inference completed. Results saved to {args.output}")


    elif args.task == "create_data":
        # Call the create_data function
        create_data(file_path=args.input_file, out_path=args.output_file)
        print(f"Synthetic data created. Saved to {args.output_file}")

    elif args.task == "expand_dataset":
        # expand the dataset / change to format where each row corresponds to one customer
        preprocess_and_expand(
            input_file=args.input_file_changeformat,
            output_file=args.output_file_changeformat,
            expansion_factor=args.expansion_factor,
            embedding_lookup=args.lookup_file
            )
        print(f"Data format changed. Saved to {args.output_file_changeformat}")

    elif args.task == "train_reward_function":

        if not args.embedding_lookup_path:
            raise ValueError("--embedding_lookup_path must be specified for ID-based training.")
        
        if not args.train_data_path:
            raise ValueError("--train_data_path must be specified for training.")

        print("Loading embedding lookup...")
        embedding_lookup_df = pd.read_parquet(args.embedding_lookup_path)

        feature_columns = [
            "aesthetic_score", "ratings", "discount_percentage", "discount_price_log", "actual_price_log"
        ]

        categorical_cols = [
            "main_category", "sub_category", "gender", "age_group"
        ]

        feature_sizes_path = "feature_sizes.json"

        if os.path.exists(feature_sizes_path):
            print(f"Loading cached feature sizes from {feature_sizes_path}...")
            with open(feature_sizes_path, "r") as f:
                feature_sizes = json.load(f)
        else:
            print("Computing feature sizes dynamically from training data...")
            train_df = pd.read_parquet(args.train_data_path)
            feature_sizes = {col: int(train_df[col].nunique()) for col in categorical_cols}

            print(f"Saving computed feature sizes to {feature_sizes_path}...")
            with open(feature_sizes_path, "w") as f:
                json.dump(feature_sizes, f)

        # Ensure feature_sizes is a list in correct order
        feature_sizes_list = [19, 110, 2, 7, 1, 1, 1, 1, 1]# ones for continous features#[feature_sizes[col] for col in categorical_cols]

        print(f"Feature sizes: {feature_sizes_list}")

        text_dim = 768  # or: int(np.load("train_data_text_dim.npy")[0])
        image_dim = 768

        print("Preparing training dataset...")
        dataset = ParquetDeepFMDatasetWithLookup(
            data_path=args.train_data_path,
            embedding_lookup_df=embedding_lookup_df,
            feature_columns=feature_columns,
            categorical_cols=categorical_cols,
            text_id_col="clip_text_embedding_id",
            image_id_col="clip_embedding_id",
            label_col="clicked"
        )

        #dataloader = DataLoader(
        #    dataset,
        #    batch_size=512,
        #    shuffle=True,
        #    num_workers=4,
        #    pin_memory=True
        #)

        # Define fusion module
        fusion_module = MLPFusion(
            clip_dim=text_dim,  # assuming both text/image are same size
            demo_emb_dim=8,
            joint_dim=64
        )

        # Train
        print("Starting training...")
        #model, loss_hist = train_deepfm_from_dataloader(
        #    dataloader=dataloader,
        #    feature_sizes=feature_sizes_list,
        #    epochs=args.epochs,
        #    lr=args.lr,
        #    fusion_module=fusion_module,
        #    fusion_input_dim=text_dim + image_dim
        #)

        # Optionally load demographic regularizers
        dem_regularizers = None
        if args.dem_probe_path:
            assert len(args.dem_probe_path) == len(args.lambda_dem) == len(args.dem_label_index), \
                "Demographic regularizer arguments must have the same length"

            dem_regularizers = []
            print("Demographic Regularizers:")
            for path, lam, idx in zip(args.dem_probe_path, args.lambda_dem, args.dem_label_index):
                print(f"  - {path} (λ={lam}, index={idx})")
                reg = DemographicRegularizer(probe_path=path, lambda_reg=lam)
                dem_regularizers.append((reg, idx))

        model, metrics = train_deepfm_with_validation(
            dataset=dataset,
            feature_sizes=feature_sizes_list,
            epochs=args.epochs,
            lr=args.lr,
            #fusion_module=fusion_module,
            #fusion_input_dim=text_dim + image_dim,
            batch_size=args.batch_size_training,
            val_split=0.1,
            dropout=[args.dropout, args.dropout, args.dropout],
            embedding_size=args.embedding_size,
            shape=args.shape,
            dem_regularizers=dem_regularizers
        )

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(args.loss_out, index=False)

        # Save model and training history
        torch.save(model.state_dict(), args.model_out)
        #pd.DataFrame({"loss": loss_hist}).to_csv(args.loss_out, index=False)

        print(f"Model saved to {args.model_out}")
        print(f"Loss history saved to {args.loss_out}")

    elif args.task == "evaluate":
        evaluate_deepfm(
            model_path=args.model_out,
            data_path=args.test_data_path,
            embedding_lookup_path=args.embedding_lookup_path,
            feature_columns=[
                "aesthetic_score", "ratings", "discount_percentage",
                "discount_price_log", "actual_price_log"
            ],
            categorical_cols=["main_category", "sub_category", "gender", "age_group"],
            fusion_input_dim=1536,  # 768 text + 768 image
            embedding_size=args.embedding_size,
            shape=args.shape,
            csv_path=f"metrics_eval_{args.model_out}.csv",
            json_path=f"metrics_eval_{args.model_out}.json"
        )

    elif args.task == "train_clip_gender_probe":
        embedding_lookup_df = pd.read_parquet(args.embedding_lookup_path)

        dataset = ClipEmbeddingGenderDataset(
            data_path=args.train_data_path,
            embedding_lookup_df=embedding_lookup_df,
            embedding_type="image",  # or "text"
            embedding_col="clip_embedding",  # or "clip_text_embedding"
            label_col="gender"
        )

        train_logreg_on_clip_embedding_gender_probe(
            dataset=dataset,
            use_text_embed=(args.embedding_type == "text"),
            batch_size=args.batch_size_training,
            epochs=args.epochs,
            model_out="logreg_gender_probe.pkl"
        )

    elif args.task == "train_clip_age_probe":
        embedding_lookup_df = pd.read_parquet(args.embedding_lookup_path)
        dataset = ClipEmbeddingGenderDataset(
            data_path=args.train_data_path,
            embedding_lookup_df=embedding_lookup_df,
            embedding_type=args.embedding_type,
            embedding_col="clip_embedding",
            label_col="age_group",
            aggregate_age=False
        )
        train_logreg_on_clip_embedding_age_probe(
            dataset=dataset,
            use_text_embed=(args.embedding_type == "text"),
            batch_size=args.batch_size_training,
            epochs=args.epochs,
            model_out="logreg_age_probe.pkl"
        )

    elif args.task == "train_clip_age_probe_aggregated":
        embedding_lookup_df = pd.read_parquet(args.embedding_lookup_path)
        dataset = ClipEmbeddingGenderDataset(
            data_path=args.train_data_path,
            embedding_lookup_df=embedding_lookup_df,
            embedding_type=args.embedding_type,
            embedding_col="clip_embedding",
            label_col="age_group",
            aggregate_age=True
        )
        train_logreg_on_clip_embedding_age_probe(
            dataset=dataset,
            use_text_embed=(args.embedding_type == "text"),
            batch_size=args.batch_size_training,
            epochs=args.epochs,
            model_out=args.model_out,
            onevsrest=args.onevsrest
        )

    elif args.task == "generate_images_embeddings":
        if not args.prompt_file or not args.output_baseline:
            raise ValueError("--prompt_file and --output_baseline must be provided for prompt_only_baseline task.")
        generate_images_embeddings(args.prompt_file, args.output_baseline)

    elif args.task == "get_image_rewards":
        compute_deepfm_reward(
            data_path=args.output_baseline,
            model_path=args.model_out,
            output_path=args.output_rewards
        )

    elif args.task == "rl_finetune":
        if not args.prompt_file or not args.model_out:
            raise ValueError("You must specify --prompt_file and --deepfm_model")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading models...")
        pipe = load_pipeline_and_models(device="cuda" if torch.cuda.is_available() else "cpu")
        pipe.unet.enable_gradient_checkpointing()

        print("Loading prompts...")
        df_prompts = pd.read_parquet(args.prompt_file)

        # Load reward queries
        df_reward_queries = pd.read_parquet("df_evaluation_normalized.parquet").head(500)  # replace with your path

        reward_fn = make_deepfm_reward_function(args.model_out, device)

        print("Starting RL fine-tuning...")
        rl_finetune_loop(
            pipe=pipe,
            df_prompts=df_prompts,
            df_queries=df_reward_queries,
            reward_model_fn=reward_fn,
            num_epochs=args.epochs,
            device=device,
            rl_method=args.rl_method,
            use_kl_reg=args.use_kl_reg
        )


if __name__ == "__main__":
    main()