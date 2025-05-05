# --- Import libraries ---
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import copy

# --- Reproducibility settings ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Paths and device ---
DATA_DIR = r"C:\Users\loris\OneDrive\ETH\Group Project"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define SVD++ model ---
class SVDpp(nn.Module):
    def __init__(self, num_scientists=10000, num_papers=1000, emb_dim=64, s2p=None, global_mean=3.82):
        super().__init__()
        self.emb_dim = emb_dim
        self.s2p = s2p or {}

        # Latent embeddings for explicit and implicit interactions
        self.scientist_factors = nn.Embedding(num_scientists, emb_dim)
        self.paper_factors = nn.Embedding(num_papers, emb_dim)
        self.implicit_factors = nn.Embedding(num_papers, emb_dim)

        # Bias terms
        self.scientist_bias = nn.Embedding(num_scientists, 1)
        self.paper_bias = nn.Embedding(num_papers, 1)
        self.global_bias = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)

        # Weight initialization
        nn.init.normal_(self.scientist_factors.weight, std=0.1)
        nn.init.normal_(self.paper_factors.weight, std=0.1)
        nn.init.normal_(self.implicit_factors.weight, std=0.1)
        nn.init.constant_(self.scientist_bias.weight, 0.0)
        nn.init.constant_(self.paper_bias.weight, 0.0)

    def forward(self, scientist_ids, paper_ids):
        # Gather embeddings and biases
        scientist_embeddings = self.scientist_factors(scientist_ids)
        paper_embeddings = self.paper_factors(paper_ids)
        scientist_biases = self.scientist_bias(scientist_ids).squeeze()
        paper_biases = self.paper_bias(paper_ids).squeeze()

        # Implicit feedback vector per scientist
        papers = [self.s2p.get(k.item(), []) for k in scientist_ids]
        implicit_embeds = []
        for sp in papers:
            if sp:
                y_j = self.implicit_factors(torch.tensor(sp, device=scientist_ids.device))
                sum_yj = y_j.sum(dim=0)
                norm_yj = sum_yj / torch.sqrt(torch.tensor(len(sp), dtype=torch.float, device=scientist_ids.device))
            else:
                norm_yj = torch.zeros_like(scientist_embeddings[0])
            implicit_embeds.append(norm_yj)
        y_u = torch.stack(implicit_embeds)

        # Final prediction
        interaction = ((scientist_embeddings + y_u) * paper_embeddings).sum(dim=1)
        return interaction + scientist_biases + paper_biases + self.global_bias


# --- Simple dot-product embedding model ---
class EmbeddingDotProductModel(nn.Module):
    def __init__(self, num_scientists, num_papers, dim):
        super().__init__()
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)

    def forward(self, sid, pid):
        return torch.sum(self.scientist_emb(sid) * self.paper_emb(pid), dim=-1)


# --- Data loading and preprocessing ---
def read_data_df():
    # Load and format rating and wishlist data
    df = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"))
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df['source'] = 'explicit'

    wishlist_df = pd.read_csv(os.path.join(DATA_DIR, "train_tbr.csv"))
    wishlist_df["sid"] = wishlist_df["sid"].astype(int)
    wishlist_df["pid"] = wishlist_df["pid"].astype(int)
    wishlist_df["rating"] = 3.5
    wishlist_df['source'] = 'wishlist'

    df = pd.concat([df, wishlist_df], ignore_index=True).drop_duplicates(subset=['sid', 'pid'], keep='first')
    df['weight'] = df['source'].map({'explicit': 1.0, 'wishlist': 0.5})

    # Split for validation and create lookup dictionary
    real_ratings = df[df['source'] == 'explicit']
    train_df, valid_df = train_test_split(real_ratings, test_size=0.25, random_state=SEED)
    global_mean = torch.tensor(np.mean(train_df.rating.values), dtype=torch.float32)
    scientist2papers = df.groupby("sid")["pid"].apply(list).to_dict()
    return df, train_df, valid_df, scientist2papers, global_mean


def get_dataset_training(df):
    # Dataset with weights
    return torch.utils.data.TensorDataset(
        torch.tensor(df["sid"].values),
        torch.tensor(df["pid"].values),
        torch.tensor(df["rating"].values).float(),
        torch.tensor(df["weight"].values).float(),
    )


def get_dataset(df):
    # Dataset without weights (for validation)
    return torch.utils.data.TensorDataset(
        torch.tensor(df["sid"].values),
        torch.tensor(df["pid"].values),
        torch.tensor(df["rating"].values).float(),
    )


def evaluate(valid_df, pred_fn):
    # Compute RMSE on validation set
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)


def make_submission(pred_fn, filename):
    # Create Kaggle submission file
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0].astype(int).values
    pids = sid_pid[1].astype(int).values
    df["rating"] = pred_fn(sids, pids)
    df.to_csv(filename, index=False)


# --- Load data ---
full_df, train_df, valid_df, scientist2papers, global_mean = read_data_df()

# --- Initialize models and optimizers ---
svd_model = SVDpp(emb_dim=64, s2p=scientist2papers, global_mean=global_mean).to(device)
emb_model = EmbeddingDotProductModel(10_000, 1_000, 32).to(device)

svd_optim = torch.optim.Adam(svd_model.parameters(), lr=1e-3, weight_decay=1e-4)
emb_optim = torch.optim.Adam(emb_model.parameters(), lr=1e-3)

# --- Create dataloaders ---
train_loader = torch.utils.data.DataLoader(get_dataset_training(full_df), batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(get_dataset(valid_df), batch_size=64)

# --- Triplet sampling for contrastive learning ---
def generate_triplets(df, num_neg=1):
    grouped = df.groupby("sid")
    triplets = []
    paper_set = set(df["pid"].unique())

    for sid, group in grouped:
        pos_papers = group["pid"].tolist()
        neg_candidates = list(paper_set - set(pos_papers))
        if not neg_candidates:
            continue
        for pos in pos_papers:
            for _ in range(num_neg):
                neg = np.random.choice(neg_candidates)
                triplets.append((sid, pos, neg))

    sids, pos, neg = zip(*triplets)
    return torch.tensor(sids), torch.tensor(pos), torch.tensor(neg)

s_triplets, p_triplets_pos, p_triplets_neg = generate_triplets(train_df)
triplet_dataset = torch.utils.data.TensorDataset(s_triplets, p_triplets_pos, p_triplets_neg)
triplet_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=64, shuffle=True)

# --- Contrastive loss function ---
def contrastive_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return torch.mean(F.relu(pos_dist - neg_dist + margin))


# --- Train embedding model with contrastive loss ---
alpha = 0.1  # weight for contrastive term
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    emb_model.train()
    total_loss = 0
    total = 0
    for sid, pid, rating, _ in train_loader:
        sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)

        # MSE prediction loss
        pred = emb_model(sid, pid)
        mse = F.mse_loss(pred, rating)

        # Contrastive loss via batch-shifted negatives
        anchor = emb_model.scientist_emb(sid)
        positive = emb_model.paper_emb(pid)
        negative = torch.roll(positive, shifts=1, dims=0)
        contrastive = contrastive_loss(anchor, positive, negative)

        loss = mse + alpha * contrastive

        emb_optim.zero_grad()
        loss.backward()
        emb_optim.step()

        total_loss += loss.item() * len(sid)
        total += len(sid)

    print(f"[Embedding + Contrastive Epoch {epoch+1}] RMSE: {(total_loss / total)**0.5:.4f}")


# --- Train SVD++ with early stopping ---
best_rmse = float("inf")
patience = 2
epochs_no_improve = 0
best_model_state = None

for epoch in range(10):
    svd_model.train()
    total_loss = 0
    total = 0
    for sid, pid, rating, weight in train_loader:
        sid, pid, rating, weight = sid.to(device), pid.to(device), rating.to(device), weight.to(device)
        pred = svd_model(sid, pid)
        loss = (weight * (pred - rating) ** 2).mean()

        svd_optim.zero_grad()
        loss.backward()
        svd_optim.step()

        total_loss += loss.item() * len(sid)
        total += len(sid)

    # Validation RMSE
    svd_model.eval()
    total_val_loss = 0
    total_val_count = 0
    for sid, pid, rating in valid_loader:
        sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)
        with torch.no_grad():
            pred = svd_model(sid, pid).clamp(1, 5)
            loss = F.mse_loss(pred, rating)
            total_val_loss += loss.item() * len(sid)
            total_val_count += len(sid)

    val_rmse = (total_val_loss / total_val_count) ** 0.5
    print(f"[SVD++ Epoch {epoch+1}] Train RMSE: {(total_loss / total)**0.5:.4f}, Validation RMSE: {val_rmse:.4f}")

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_model_state = copy.deepcopy(svd_model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Stopped early at epoch {epoch+1}. Best RMSE: {best_rmse:.4f}")
            break

svd_model.load_state_dict(best_model_state)


# --- Final prediction wrappers ---
def emb_pred_fn(sids, pids):
    with torch.no_grad():
        return emb_model(torch.tensor(sids).to(device), torch.tensor(pids).to(device)).clamp(1, 5).cpu().numpy()

def svd_pred_fn(sids, pids):
    with torch.no_grad():
        return svd_model(torch.tensor(sids).to(device), torch.tensor(pids).to(device)).clamp(1, 5).cpu().numpy()

def ensemble_pred_fn(sids, pids):
    with torch.no_grad():
        s = torch.tensor(sids).to(device)
        p = torch.tensor(pids).to(device)
        emb = emb_model(s, p)
        svd = svd_model(s, p)
        return (0.5 * emb + 0.5 * svd).clamp(1, 5).cpu().numpy()

# --- Final scores and submission ---
print("\n--- Final RMSE Scores ---")
print(f"Embedding Only RMSE: {evaluate(valid_df, emb_pred_fn):.4f}")
print(f"SVD++ Only RMSE: {evaluate(valid_df, svd_pred_fn):.4f}")
print(f"Ensemble RMSE: {evaluate(valid_df, ensemble_pred_fn):.4f}")

make_submission(ensemble_pred_fn, "ensemble_submission.csv")