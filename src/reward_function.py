import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from tqdm import tqdm
import time
import json
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.multiclass import OneVsRestClassifier

# MLPFusion class
class MLPFusion(nn.Module):
    def __init__(self, clip_dim=768, demo_emb_dim=8, joint_dim=64, num_age_groups=7, dropout_rate=0.3):
        super().__init__()

        self.output_dim = joint_dim

        self.gender_embedding = nn.Embedding(2, demo_emb_dim)
        self.age_embedding = nn.Embedding(num_age_groups, demo_emb_dim)

        input_dim = 2 * clip_dim + 2 * demo_emb_dim  # text + image + gender + age

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, text_embed, image_embed, gender_idx, age_idx):
        g_emb = self.gender_embedding(gender_idx)
        a_emb = self.age_embedding(age_idx)
        x = torch.cat([text_embed, image_embed, g_emb, a_emb], dim=1)
        joint_embed = self.fusion(x)
        return joint_embed

# DeepFM Class
class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=8, hidden_dims=[800, 800, 800], dropout=[0.5, 0.5, 0.5], fusion_module=None, fusion_input_dim=768):
        super(DeepFM, self).__init__()
        self.field_size = len(feature_sizes)
        self.embedding_size = embedding_size
        self.fusion_module = fusion_module
        self.fusion_input_dim = fusion_input_dim

        # FM parts
        self.fm_first_order = nn.ModuleList([
            nn.Embedding(size, 1) for size in feature_sizes
        ])
        self.fm_second_order = nn.ModuleList([
            nn.Embedding(size, embedding_size) for size in feature_sizes
        ])

        # Deep part
        # Deep part input size
        base_deep_input_dim = self.field_size * embedding_size
        total_deep_input_dim = base_deep_input_dim + (fusion_module.output_dim if fusion_module else 0)

        all_dims = [total_deep_input_dim] + hidden_dims + [1]
        self.deep_layers = nn.ModuleList([
            nn.Linear(all_dims[i], all_dims[i+1]) for i in range(len(all_dims)-1)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(p) for p in dropout
        ])
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(1))

    def cache_embeddings(self, text_embed, image_embed):
        self._cached_text_embed = text_embed
        self._cached_image_embed = image_embed

    def forward(self, Xi, Xv):
        # First-order term
        first_order = [emb(Xi[:, i]) * Xv[:, i].unsqueeze(1) for i, emb in enumerate(self.fm_first_order)]
        first_order_sum = torch.sum(torch.cat(first_order, dim=1), dim=1)

        # Second-order term
        embeds = [emb(Xi[:, i]) * Xv[:, i].unsqueeze(1) for i, emb in enumerate(self.fm_second_order)]
        summed = sum(embeds)
        summed_square = summed * summed
        square_sum = sum([x * x for x in embeds])
        second_order_sum = 0.5 * torch.sum(summed_square - square_sum, dim=1)

        # Deep input from FM embeddings
        deep_input = torch.cat(embeds, dim=1)

        # Optional: add fusion output to deep input
        if self.fusion_module is not None:
            gender_idx = Xi[:, 2]  # gender index
            age_idx = Xi[:, 3]     # age index
            #fusion_input = Xv[:, -self.fusion_input_dim:]         # shape (batch, 1536)
            #half = self.fusion_input_dim // 2                     # 768
            #text_embed = fusion_input[:, :half]                   # (batch, 768)
            #image_embed = fusion_input[:, half:]                  # (batch, 768)
            
            fusion_out = self.fusion_module(self._cached_text_embed, self._cached_image_embed, gender_idx, age_idx)
            deep_input = torch.cat([deep_input, fusion_out], dim=1)

        # Deep forward pass
        for i, layer in enumerate(self.deep_layers[:-1]):
            deep_input = self.relu(layer(deep_input))
            if i < len(self.dropouts):
                deep_input = self.dropouts[i](deep_input)
        deep_output = self.deep_layers[-1](deep_input).squeeze(1)

        # Final output
        return first_order_sum + second_order_sum + deep_output + self.bias
    
class ParquetDeepFMDatasetWithLookup(torch.utils.data.Dataset):
    def __init__(self, data_path, embedding_lookup_df, feature_columns, categorical_cols,
                 text_id_col="clip_text_embedding_id", image_id_col="clip_embedding_id", label_col="clicked", embed_continuous_as_fm=True):
        self.df = pd.read_parquet(data_path)
        self.embeddings = embedding_lookup_df.set_index("id")
        self.feature_columns = feature_columns
        self.categorical_cols = categorical_cols
        self.text_id_col = text_id_col
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.embed_continuous_as_fm = embed_continuous_as_fm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract Xi (categorical indices)
        Xi_cat = torch.tensor([int(row[col]) for col in self.categorical_cols], dtype=torch.long)

        # Xv = ones (same shape as Xi), as per original DeepFM formulation
        Xv_cat = torch.ones_like(Xi_cat, dtype=torch.float32)

        # Structured features
        X_struct = np.array([row[col] for col in self.feature_columns], dtype=np.float32)

        if self.embed_continuous_as_fm:
            Xi_cont = torch.zeros(len(self.feature_columns), dtype=torch.long)  # dummy indices
            Xv_cont = torch.tensor(X_struct, dtype=torch.float32)

            Xi = torch.cat([Xi_cat, Xi_cont], dim=0)
            Xv = torch.cat([Xv_cat, Xv_cont], dim=0)
        else:
            Xi = Xi_cat
            Xv = Xv_cat

        # Embedding lookup by ID
        text_embed = self.embeddings.loc[row[self.text_id_col], "clip_text_embedding"]
        image_embed = self.embeddings.loc[row[self.image_id_col], "clip_embedding"]

        text_embed = np.array(text_embed, dtype=np.float32)
        image_embed = np.array(image_embed, dtype=np.float32)

        # Deep input = structured + text + image
        deep_input = np.concatenate([text_embed, image_embed], dtype=np.float32)

        y = torch.tensor(row[self.label_col], dtype=torch.float32)

        return Xi, Xv, torch.tensor(deep_input), y
    
class ParquetDeepFMDatasetDirectEmbed(torch.utils.data.Dataset):
    def __init__(self, df, feature_columns, categorical_cols,
                 text_embed_col="clip_text_embedding", image_embed_col="clip_embedding",
                 label_col=None, embed_continuous_as_fm=True):
        self.df = df
        self.feature_columns = feature_columns
        self.categorical_cols = categorical_cols
        self.text_embed_col = text_embed_col
        self.image_embed_col = image_embed_col
        self.label_col = label_col
        self.embed_continuous_as_fm = embed_continuous_as_fm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Categorical features
        Xi_cat = torch.tensor([int(row[col]) for col in self.categorical_cols], dtype=torch.long)
        Xv_cat = torch.ones_like(Xi_cat, dtype=torch.float32)

        # Continuous features
        X_struct = np.array([row[col] for col in self.feature_columns], dtype=np.float32)

        if self.embed_continuous_as_fm:
            Xi_cont = torch.zeros(len(self.feature_columns), dtype=torch.long)
            Xv_cont = torch.tensor(X_struct, dtype=torch.float32)
            Xi = torch.cat([Xi_cat, Xi_cont], dim=0)
            Xv = torch.cat([Xv_cat, Xv_cont], dim=0)
        else:
            Xi = Xi_cat
            Xv = Xv_cat

        #text_embed = np.array(row[self.text_embed_col], dtype=np.float32)
        #image_embed = np.array(row[self.image_embed_col], dtype=np.float32)
        #deep_input = np.concatenate([text_embed, image_embed], dtype=np.float32)

        return Xi, Xv#, torch.tensor(deep_input)

def train_deepfm_with_validation(
    dataset,
    feature_sizes,
    epochs=100,
    lr=1e-3,
    fusion_module=None,
    fusion_input_dim=768,
    device=None,
    batch_size=512,
    val_split=0.1,
    dropout=[0.5, 0.5, 0.5],
    embedding_size=8,
    shape=[800, 800, 800],
    dem_regularizers=None # list of tuples: (DemographicRegularizer, label_index)
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split into training and validation datasets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Model
    model = DeepFM(
        feature_sizes=feature_sizes,
        embedding_size=embedding_size,
        hidden_dims=shape,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loss_history = []
    val_loss_history = []
    val_auc_history = []
    val_accuracy_history = []
    val_f1_history = []
    reg_loss_history = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_reg_loss_total = 0.0
        start_time = time.time()

        for Xi, Xv, deep_input, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            Xi, Xv, deep_input, y = Xi.to(device), Xv.to(device), deep_input.to(device), y.to(device)
            
            text_embed = deep_input[:, :768]
            image_embed = deep_input[:, 768:1536]

            model.cache_embeddings(text_embed, image_embed)

            optimizer.zero_grad()
            logits = model(Xi, Xv).squeeze()
            loss = criterion(logits, y)

            # --- Add Demographic Regularization ---
            batch_reg_loss = 0.0
            if dem_regularizers is not None:
                for reg, label_index in dem_regularizers:
                    labels = Xi[:, label_index]

                    # Special case: age group (7-class to 3-class mapping)
                    if reg.reg_type == "multiclass":
                        labels = labels.long().contiguous()
                        labels = torch.bucketize(labels, boundaries=torch.tensor([2, 4]).to(labels.device))

                    reg_loss = reg.compute_loss(image_embed, labels)
                    batch_reg_loss += reg_loss

                loss += batch_reg_loss
                epoch_reg_loss_total += batch_reg_loss.item()

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        avg_reg_loss = epoch_reg_loss_total / len(train_loader)
        reg_loss_history.append(avg_reg_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for Xi_val, Xv_val, deep_input_val, y_val in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                Xi_val, Xv_val, deep_input_val, y_val = Xi_val.to(device), Xv_val.to(device), deep_input_val.to(device), y_val.to(device)

                text_embed = deep_input_val[:, :768]
                image_embed = deep_input_val[:, 768:1536]

                model.cache_embeddings(text_embed, image_embed)

                logits = model(Xi_val, Xv_val).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                #preds = (probs > 0.5).astype(int)

                loss = criterion(logits, y_val)
                total_val_loss += loss.item()

                all_preds.extend(probs)
                all_labels.extend(y_val.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Compute metrics
        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        val_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        val_auc_history.append(val_auc)
        val_accuracy_history.append(val_accuracy)
        val_f1_history.append(val_f1)

        # Log the epoch
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Reg Loss:   {avg_reg_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")
        print(f"  Val Acc:    {val_accuracy:.4f}")
        print(f"  Val F1:     {val_f1:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f} seconds\n")

    return model, {
        "train_loss": train_loss_history,
        "reg_loss": reg_loss_history,
        "val_loss": val_loss_history,
        "val_auc": val_auc_history,
        "val_accuracy": val_accuracy_history,
        "val_f1": val_f1_history
    }

def evaluate_deepfm(
    model_path,
    data_path,
    embedding_lookup_path,
    feature_columns,
    categorical_cols,
    fusion_input_dim,
    fusion_hidden_dim=64,
    demo_emb_dim=8,
    batch_size=2048,
    embedding_size=8,
    shape=[800, 800, 800],
    csv_path=None,
    json_path=None
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load embedding lookup
    embedding_lookup_df = pd.read_parquet(embedding_lookup_path)

    feature_sizes_list = [19, 110, 2, 7, 1, 1, 1, 1, 1]

    # Prepare dataset
    dataset = ParquetDeepFMDatasetWithLookup(
        data_path=data_path,
        embedding_lookup_df=embedding_lookup_df,
        feature_columns=feature_columns,
        categorical_cols=categorical_cols,
        text_id_col="clip_text_embedding_id",
        image_id_col="clip_embedding_id",
        label_col="clicked"
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = DeepFM(
        feature_sizes=feature_sizes_list,
        embedding_size=embedding_size,
        hidden_dims=shape
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run evaluation
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for Xi, Xv, deep_input, y in tqdm(dataloader, desc="Evaluating"):
            Xi = Xi.to(device)
            Xv = Xv.to(device)
            deep_input = deep_input.to(device)
            y = y.to(device)

            text_embed = deep_input[:, :768]
            image_embed = deep_input[:, 768:1536]
            model.cache_embeddings(text_embed, image_embed)

            logits = model(Xi, Xv).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_probs > 0.5).astype(int))
    loss = log_loss(all_labels, all_probs)

    results = {
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "logloss": round(loss, 4)
    }

    print("\n".join(f"{k.capitalize()}: {v}" for k, v in results.items()))

    if csv_path is not None:
        pd.DataFrame([results]).to_csv(csv_path, index=False)

    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

class ClipEmbeddingGenderDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, embedding_lookup_df, embedding_type="image", embedding_col="clip_embedding", label_col="gender", embedding_id_col=None, aggregate_age=False):
        """
        Args:
            data_path: path to .parquet file with metadata
            embedding_lookup_df: DataFrame with 'id' and embedding columns
            embedding_type: "image" or "text"
            embedding_col: column name in embedding_lookup_df that contains the vector
            label_col: column in data_path indicating gender (0/1)
            embedding_id_col: column in data_path to lookup embeddings by ID
            aggregate_age: if True, will aggregate age groups into 3 categories
        """
        self.df = pd.read_parquet(data_path)

        if aggregate_age and label_col == "age_group":
            self.df[label_col] = self.df[label_col].map({
                0: 0, 1: 0,     # 18–34
                2: 1, 3: 1,     # 35–54
                4: 2, 5: 2, 6: 2  # 55+
            })

        self.embeddings = embedding_lookup_df.set_index("id")
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.embedding_id_col = embedding_id_col or ("clip_embedding_id" if embedding_type == "image" else "clip_text_embedding_id")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embed_id = row[self.embedding_id_col]
        embedding = np.array(self.embeddings.loc[embed_id][self.embedding_col], dtype=np.float32)
        label = int(row[self.label_col])
        return torch.tensor(embedding), torch.tensor(label, dtype=torch.float32)

def train_logreg_on_clip_embedding_gender_probe(dataset, use_text_embed=False, batch_size=1024, epochs=1, val_split=0.1, model_out="logreg_gender_probe.pkl"):
    # Prepare train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    first_batch = True

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for embedding, gender in tqdm(train_loader, desc="Training"):
            gender = gender.numpy()

            X = embedding.numpy()

            if first_batch:
                model.partial_fit(X, gender, classes=np.array([0, 1]))
                first_batch = False
            else:
                model.partial_fit(X, gender)

    # Validation
    all_preds, all_labels = [], []
    for embedding, gender in tqdm(val_loader, desc="Validation"):
        y_val = gender.numpy()
        X_val = embedding.numpy()
        probs = model.predict_proba(X_val)[:, 1]
        all_preds.extend(probs)
        all_labels.extend(y_val)

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    logloss = log_loss(all_labels, all_preds)

    print(f"Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}, LogLoss: {logloss:.4f}")
    joblib.dump(model, model_out)
    print(f"Saved gender probe model to: {model_out}")

def train_logreg_on_clip_embedding_age_probe(dataset, use_text_embed=False, batch_size=1024, epochs=1, val_split=0.1, model_out="logreg_age_probe.pkl", onevsrest=False):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    base_model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    model = OneVsRestClassifier(base_model) if onevsrest else base_model
    first_batch = True

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for embedding, age in tqdm(train_loader, desc="Training"):
            age = age.numpy()
            X = embedding.numpy()
            if first_batch:
                classes = np.unique(age)
                model.partial_fit(X, age, classes=classes)
                first_batch = False
            else:
                model.partial_fit(X, age)

    all_preds, all_probs, all_labels = [], [], []
    for embedding, age in tqdm(val_loader, desc="Validation"):
        y_val = age.numpy()
        X_val = embedding.numpy()
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y_val)

    acc = accuracy_score(all_labels, np.array(all_preds))
    auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
    logloss = log_loss(all_labels, np.array(all_probs))

    print(f"Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}, LogLoss: {logloss:.4f}")
    joblib.dump(model, model_out)
    print(f"Saved age probe model to: {model_out}")

# --- Demographic Regularization Module ---
class DemographicRegularizer:
    def __init__(self, probe_path, lambda_reg=1.0):
        self.model = joblib.load(probe_path)
        self.lambda_reg = lambda_reg

        if hasattr(self.model, "estimators_"):
            # One-vs-rest multiclass case
            coef = np.vstack([clf.coef_ for clf in self.model.estimators_])
            W = coef / np.linalg.norm(coef, axis=1, keepdims=True)
            self.coef_matrix = torch.tensor(W, dtype=torch.float32)
            self.reg_type = "multiclass"
        else:
            # Binary case
            coef = self.model.coef_[0]
            w = coef / np.linalg.norm(coef)
            self.coef_vector = torch.tensor(w, dtype=torch.float32)
            self.reg_type = "binary"

    def compute_loss(self, embeddings, labels):
        # embeddings: (B, D), labels: (B,)
        embeddings = F.normalize(embeddings, dim=1)  # Normalize z

        if self.reg_type == "multiclass":
            # OvR: select direction vector for each sample
            W = self.coef_matrix.to(embeddings.device)  # (C, D)
            w_y = W[labels]  # (B, D)
            #print(f"reg type: {self.reg_type}, labels max: {labels.max().item()}, shape of W: {W.shape}")
            cos_sim = torch.sum(embeddings * w_y, dim=1)  # (B,)
            loss = -cos_sim.mean()

        elif self.reg_type == "binary":
            w = self.coef_vector.to(embeddings.device)  # (D,)
            #print(f"reg type: {self.reg_type}, labels max: {labels.max().item()}")
            cos_sim = torch.matmul(embeddings, w)  # (B,)
            target = 2 * labels.float() - 1  # map {0,1} → {−1,+1}
            loss = -(target * cos_sim).mean()

        return self.lambda_reg * loss