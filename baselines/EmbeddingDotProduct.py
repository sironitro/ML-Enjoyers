import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import copy



SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingDotProductModel(nn.Module):
    """
    A simple collaborative filtering model using dot product of embeddings.
    """
    def __init__(self, num_scientists: int, num_papers: int, dim: int):
        """
        Initializes the embedding layers for scientists and papers.

        Input:
        num_scientists (int): Total number of scientists.
        num_papers (int): Total number of papers.
        dim (int): Dimensionality of the embedding vectors.
        """
        super().__init__()

        # Assign to each scientist and paper an embedding
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        

    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Computes the predicted scores via dot product between scientist and paper embeddings.

        Input:
        sid (Tensor): Tensor of scientist IDs, shape [B].
        pid (Tensor): Tensor of paper IDs, shape [B].

        Outputs:
        Tensor: Predicted scores for each (scientist, paper) pair, shape [B].
        """
        # Per-pair dot product and sum across embedding dimensions
        return torch.sum(self.scientist_emb(sid) * self.paper_emb(pid), dim=-1)
    
    

# Define model (10k scientists, 1k papers, 32-dimensional embeddings) and optimizer
model = EmbeddingDotProductModel(10_000, 1_000, 32).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Get train and validation data and setup data loaders
train_df, valid_df, _ = read_data_df()
train_dataset = get_dataset(train_df)
valid_dataset = get_dataset(valid_df)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

best_rmse = float("inf")
patience = 2
epochs_no_improve = 0
best_model_state = None
NUM_EPOCHS = 300

# Training loop
for epoch in range(NUM_EPOCHS):
    # Train model for an epoch
    total_loss = 0.0
    total_data = 0
    model.train()
    for sid, pid, ratings in train_loader:
        # Move data to GPU
        sid = sid.to(device)
        pid = pid.to(device)
        ratings = ratings.to(device)

        # Make prediction and compute loss
        pred = model(sid, pid)
        loss = F.mse_loss(pred, ratings)

        # Compute gradients w.r.t. loss and take a step in that direction
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Keep track of running loss
        total_data += len(sid)
        total_loss += len(sid) * loss.item()

    # Evaluate model on validation data
    total_val_mse = 0.0
    total_val_data = 0
    model.eval()
    for sid, pid, ratings in valid_loader:
        # Move data to GPU
        sid = sid.to(device)
        pid = pid.to(device)
        ratings = ratings.to(device)

        # Clamp predictions in [1,5], since all ground-truth ratings are
        pred = model(sid, pid).clamp(1, 5)
        mse = F.mse_loss(pred, ratings)

        # Keep track of running metrics
        total_val_data += len(sid)
        total_val_mse += len(sid) * mse.item()
    
    val_rmse = (total_val_mse / total_val_data) ** 0.5
    train_loss = total_loss / total_data
    
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train loss={total_loss / total_data:.3f}, Valid RMSE={(total_val_mse / total_val_data) ** 0.5:.3f}")
    
    # Early stopping check
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Stopped early at epoch {epoch+1}. Best RMSE: {best_rmse:.4f}")
        break


# Define prediction function
pred_fn = lambda sids, pids: model(torch.from_numpy(sids).to(device), torch.from_numpy(pids).to(device)).clamp(1, 5).cpu().numpy()

# Evaluate on validation data and get submission
with torch.no_grad():
    val_score = evaluate(valid_df, pred_fn)

print(f"Validation RMSE: {val_score:.3f}")

with torch.no_grad():
    make_submission(pred_fn, "submissions/learned_embedding_submission.csv")