import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Preprocesses the dataset by scaling, log-transforming prices, encoding categories, and dropping NaNs."""
    scaler = MinMaxScaler()
    df[['aesthetic_score', 'ratings', 'discount_percentage']] = scaler.fit_transform(
        df[['aesthetic_score', 'ratings', 'discount_percentage']]
    )
    df['discount_price_log'] = np.log1p(df['discount_price'])
    df['actual_price_log'] = np.log1p(df['actual_price'])

    main_category_mapping = {cat: idx for idx, cat in enumerate(df['main_category'].unique())}
    sub_category_mapping = {cat: idx for idx, cat in enumerate(df['sub_category'].unique())}
    df['main_category'] = df['main_category'].map(main_category_mapping)
    df['sub_category'] = df['sub_category'].map(sub_category_mapping)

    df = df.dropna()
    return df

def expand_dataset(df, expansion_factor=1000):
    base_columns = [
        "main_category", "sub_category", "aesthetic_score", "ratings",
        "discount_percentage", "discount_price_log", "actual_price_log"
    ]

    embedding_columns = ["clip_embedding", "clip_text_embedding", "bert_embedding"]
    present_embeddings = [col for col in embedding_columns if col in df.columns]

    # Create output columns: replace embedding column names with *_id
    output_columns = base_columns + [f"{col}_id" for col in present_embeddings]

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-85"]
    gender_groups = ["male", "female"]
    expanded_data = []

    for row_id, row in df.iterrows():
        for age in age_groups:
            for gender in gender_groups:
                impressions_col = f"impressions_{age}_{gender}"
                clicks_col = f"clicks_{age}_{gender}"

                if impressions_col in df.columns and row[impressions_col] > 0:
                    num_impressions = round((row[impressions_col] / row["impressions"]) * expansion_factor)
                    num_clicks = round((row[clicks_col] / row[impressions_col]) * num_impressions) if clicks_col in df.columns else 0

                    base_values = []
                    for col in base_columns:
                        base_values.append(row[col])
                    for _ in present_embeddings:
                        base_values.append(row_id)  # just the ID, not the vector

                    new_rows = [
                        base_values + [gender, age, int(i < num_clicks)]
                        for i in range(num_impressions)
                    ]
                    expanded_data.append(new_rows)

    if expanded_data:
        expanded_df = pd.DataFrame(
            [item for batch in expanded_data for item in batch],
            columns=output_columns + ["gender", "age_group", "clicked"]
        )
        gender_mapping = {"male": 0, "female": 1}
        age_mapping = {age: idx for idx, age in enumerate(age_groups)}
        expanded_df["gender"] = expanded_df["gender"].map(gender_mapping)
        expanded_df["age_group"] = expanded_df["age_group"].map(age_mapping)

        scalar_columns = base_columns + ["gender", "age_group"]
        expanded_df[scalar_columns] = expanded_df[scalar_columns].astype(float)
        expanded_df["clicked"] = expanded_df["clicked"].astype(int)

        return expanded_df
    return None

def save_embedding_lookup(df, output_file, embedding_columns=None):
    """
    Extracts embedding vectors from a parquet file and saves them in a lookup table format.

    Parameters:
        input_file (parquet): parquet with embeddings.
        output_file (str): Path to save the output parquet lookup.
        embedding_columns (list[str], optional): Columns to extract. If None, will auto-detect.
    """
    
    if embedding_columns is None:
        embedding_columns = [col for col in df.columns if col.endswith("_embedding")]

    if not embedding_columns:
        raise ValueError("No embedding columns found in the dataset.")

    # Create new DataFrame with one row per ID
    lookup_df = df[embedding_columns].copy()
    lookup_df["id"] = df.index
    lookup_df = lookup_df[["id"] + embedding_columns]

    lookup_df.to_parquet(output_file, compression="zstd", index=False)
    print(f"Embedding lookup saved to {output_file}")

def preprocess_and_expand(input_file, output_file, embedding_lookup=True, expansion_factor=1000):
    """Reads a parquet file, preprocesses and expands the dataset, then saves it to a new parquet file."""
    df = pd.read_parquet(input_file)

    # add a file with key-value pairs for the embeddings for more leightweight expanded dataset
    if embedding_lookup:
        save_embedding_lookup(df, "embedding_dict.parquet", embedding_columns=["clip_embedding", "clip_text_embedding"])

    df = preprocess_data(df)
    df_expanded = expand_dataset(df, expansion_factor=expansion_factor)
    
    if df_expanded is not None:
        df_expanded.to_parquet(output_file, index=False, compression="brotli")
        print(f"Processed dataset saved to {output_file}")
    else:
        print("No data was expanded.")