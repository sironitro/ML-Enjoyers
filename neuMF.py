# Neural Collaborative Filtering (NeuMF) Implementation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Callable, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

DATA_DIR = "./data"

def read_data_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in data and splits it into training and validation sets with a 75/25 split."""
    
    df = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"))

    # Split sid_pid into sid and pid columns
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    
    # Split into train and validation dataset
    train_df, valid_df = train_test_split(df, test_size=0.25)
    return train_df, valid_df

def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)

class RatingDataset(Dataset):
    def __init__(self, df):
        self.sids = df['sid'].values.astype(np.int64)
        self.pids = df['pid'].values.astype(np.int64)
        self.ratings = df['rating'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.sids[idx], self.pids[idx], self.ratings[idx]

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_layer_sizes=[64,32,16,8], dropout=0.2):
        super().__init__()
        # GMF embeddings
        self.user_gmf = nn.Embedding(num_users, mf_dim)
        self.item_gmf = nn.Embedding(num_items, mf_dim)
        # MLP embeddings
        self.user_mlp = nn.Embedding(num_users, mlp_layer_sizes[0])
        self.item_mlp = nn.Embedding(num_items, mlp_layer_sizes[0])
        
        # MLP layers
        mlp_layers = []
        in_size = mlp_layer_sizes[0] * 2
        for out_size in mlp_layer_sizes[1:]:
            mlp_layers += [nn.Dropout(dropout), nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction
        self.predict = nn.Linear(mf_dim + mlp_layer_sizes[-1], 1)
        
        # Initialization
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict.weight, a=1, nonlinearity='sigmoid')

    def forward(self, u, i):
        gmf = self.user_gmf(u) * self.item_gmf(i)
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=1))
        x = torch.cat([gmf, mlp], dim=1)
        return self.predict(x).squeeze()

class EnhancedNeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=64, mlp_layer_sizes=[128,64,32], dropout=0.3):
        super().__init__()
        # GMF embeddings
        self.user_gmf = nn.Embedding(num_users, mf_dim)
        self.item_gmf = nn.Embedding(num_items, mf_dim)
        # MLP embeddings
        self.user_mlp = nn.Embedding(num_users, mlp_layer_sizes[0])
        self.item_mlp = nn.Embedding(num_items, mlp_layer_sizes[0])
        
        # Add user and item bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Improved MLP with BatchNorm
        mlp_layers = []
        in_size = mlp_layer_sizes[0] * 2
        for out_size in mlp_layer_sizes[1:]:
            mlp_layers += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction
        self.predict = nn.Linear(mf_dim + mlp_layer_sizes[-1], 1)
        
        # Initialization
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict.weight, a=1, nonlinearity='sigmoid')

    def forward(self, u, i):
        # GMF path
        gmf = self.user_gmf(u) * self.item_gmf(i)
        # MLP path
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=1))
        # Combine paths
        x = torch.cat([gmf, mlp], dim=1)
        base_pred = self.predict(x).squeeze()
        # Add bias terms
        u_bias = self.user_bias(u).squeeze()
        i_bias = self.item_bias(i).squeeze()
        return base_pred + u_bias + i_bias


def preprocess_data(df):
    """Apply user-specific normalization to ratings"""
    # Get global mean
    global_mean = df['rating'].mean()
    print(f"Global mean rating: {global_mean:.4f}")
    
    # Get user biases (average rating deviation from global mean)
    user_biases = df.groupby('sid')['rating'].mean() - global_mean
    
    # Create a copy to avoid modifying the original dataframe
    df_norm = df.copy()
    
    # Normalize ratings by user bias
    def normalize_rating(row):
        user_id = row['sid']
        return row['rating'] - user_biases.get(user_id, 0)
    
    df_norm['rating'] = df_norm.apply(normalize_rating, axis=1)
    
    return df_norm, user_biases, global_mean

def train(model, train_df, valid_df, loader, optimizer, criterion, device, epochs=20):
    best_rmse = float('inf')
    
    for epoch in range(1, epochs+1):
        # Training step
        model.train()
        total_loss = 0
        
        for sids, pids, ratings in loader:
            sids, pids, ratings = sids.to(device), pids.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(sids, pids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)
            
        # Create prediction function for evaluation
        def pred_fn(s, p):
            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.from_numpy(s).to(device), 
                    torch.from_numpy(p).to(device)
                ).detach().cpu().numpy()
            return np.clip(preds, 1, 5)
        
        # Evaluate on both train and validation sets
        train_rmse = evaluate(train_df, pred_fn)
        valid_rmse = evaluate(valid_df, pred_fn)
        
        # Learning rate scheduling (if you have a scheduler)
        # scheduler.step(valid_rmse)
        
        print(f"Epoch {epoch:02d} — Train RMSE: {train_rmse:.4f}, Valid RMSE: {valid_rmse:.4f}")
        
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            torch.save(model.state_dict(), 'best_ncf.pth')
    
    print(f"\nBest Val RMSE: {best_rmse:.4f}")
    return best_rmse

def train_enhanced(model, train_df, valid_df, loader, optimizer, criterion, device, 
                  epochs=20, patience=5, clip_norm=1.0):
    best_rmse = float('inf')
    early_stop_counter = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    for epoch in range(1, epochs+1):
        # Training step
        model.train()
        total_loss = 0
        
        for sids, pids, ratings in loader:
            sids, pids, ratings = sids.to(device), pids.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(sids, pids)
            loss = criterion(preds, ratings)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            
            optimizer.step()
            total_loss += loss.item() * len(ratings)
            
        # Create prediction function for evaluation
        def pred_fn(s, p):
            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.from_numpy(s).to(device), 
                    torch.from_numpy(p).to(device)
                ).detach().cpu().numpy()
            return np.clip(preds, 1, 5)
        
        # Evaluate on both train and validation sets
        train_rmse = evaluate(train_df, pred_fn)
        valid_rmse = evaluate(valid_df, pred_fn)
        
        # Learning rate scheduling
        scheduler.step(valid_rmse)
        
        print(f"Epoch {epoch:02d} — Train RMSE: {train_rmse:.4f}, Valid RMSE: {valid_rmse:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_ncf.pth')
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    print(f"\nBest Val RMSE: {best_rmse:.4f}")
    return best_rmse

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_df, valid_df = read_data_df()
    
    # Apply preprocessing
    train_df_norm, user_biases, global_mean = preprocess_data(train_df)
    valid_df_norm = valid_df.copy()
    valid_df_norm['rating'] = valid_df_norm.apply(
        lambda row: row['rating'] - user_biases.get(row['sid'], 0), 
        axis=1
    )
    
    # Determine number of users and items
    num_users = train_df['sid'].max() + 1
    num_items = train_df['pid'].max() + 1
    print(f"Num users: {num_users}, Num items: {num_items}")
    
    # Prepare data loader with normalized data
    train_loader = DataLoader(
        RatingDataset(train_df_norm), 
        batch_size=1024, 
        shuffle=True, 
        num_workers=4
    )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use the enhanced model
    model = EnhancedNeuMF(num_users, num_items).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Train with the enhanced training function
    best_rmse = train_enhanced(
        model=model,
        train_df=train_df_norm,
        valid_df=valid_df_norm,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=20,
        patience=5
    )
