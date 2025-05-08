import copy
from model import Model
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class neuMF(Model, nn.Module):
    """
    Implementation of Neural Collaborative Filtering (NCF), combining Generalized Matrix Factorization (GMF)
    and a Multi-Layer Perceptron (MLP) into a unified architecture (NeuMF).
    The GMF captures linear latent interactions, while the MLP learns complex, nonlinear user-item relationships.
    """
    def __init__(self, name, num_users=10_000, num_items=1_000, mf_dim=64, epochs=300, mlp_layer_sizes=[128,64,32], dropout=0.3):
        """
        Initializes the NeuMF model architecture.

        Input:
            name (str): Name of the model instance.
            num_users (int): Number of user IDs.
            num_items (int): Number of item IDs.
            mf_dim (int): Dimensionality of the GMF embeddings.
            epochs (int): Number of training epochs.
            mlp_layer_sizes (list): List of MLP layer sizes.
            dropout (float): Dropout rate used in MLP layers.
        """
        nn.Module.__init__(self)
        self.to(device)
        self.name = name
        self.epochs = epochs
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
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout)
          ]
          in_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Combining GMF and MLP outputs into prediction head
        self.head = nn.Linear(mf_dim + mlp_layer_sizes[-1], 1)
        
        # Init weights
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.head.weight, a=1, nonlinearity='sigmoid')
        
        
    def forward(self, u, i):
        """
        Forward pass to compute predicted ratings.

        Input:
            u (Tensor): Tensor of user IDs.
            i (Tensor): Tensor of item IDs.

        Outputs:
            Tensor: Predicted ratings.
        """
        # GMF path
        gmf = self.user_gmf(u) * self.item_gmf(i)

        # MLP path
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=1))

        # Combine paths
        x = torch.cat([gmf, mlp], dim=1)
        base_pred = self.head(x).squeeze()

        # Add bias terms
        u_bias = self.user_bias(u).squeeze()
        i_bias = self.item_bias(i).squeeze()

        return base_pred + u_bias + i_bias
    

    def train_model(self, train_df, valid_df, lr=6e-4, weight_decay=4e-5, patience=5):
        """
        Trains the NeuMF model using MSE loss and early stopping on validation RMSE.

        Input:
            train_df (DataFrame): Training data (sid, pid, rating).
            valid_df (DataFrame): Validation data (sid, pid, rating).
            lr (float): Learning rate for Adam optimizer.
            weight_decay (float): Weight decay (L2 regularization).
            patience (int): Early stopping patience (in epochs).
        """
        self.to(device)
        best_rmse = float('inf')

        # Convert dataframes to PyTorch datasets and setup data loaders
        train_dataset = get_dataset(train_df)
        valid_dataset = get_dataset(valid_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

        # Set up Adam optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop
        for epoch in range(self.epochs):
            # Train model for an epoch
            total_loss = 0
            total_data = 0
            self.train()
            # Loop over training batches
            for sid, pid, ratings in train_loader:
                sid = sid.to(device)
                pid = pid.to(device)
                ratings = ratings.to(device)

                pred = self.forward(sid, pid)
                loss = F.mse_loss(pred, ratings)

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_data += len(sid)
                total_loss += len(sid) * loss.item()

            # Evaluate on validation set
            total_val_mse = 0.0
            total_val_data = 0
            self.eval()
            for sid, pid, ratings in valid_loader:
                sid = sid.to(device)
                pid = pid.to(device)
                ratings = ratings.to(device)

                pred = self.forward(sid, pid).clamp(1, 5)
                mse = F.mse_loss(pred, ratings)

                total_val_data += len(sid)
                total_val_mse += len(sid) * mse.item()

            val_rmse = (total_val_mse / total_val_data) ** 0.5
            train_loss = total_loss / total_data
            print(f"[Epoch {epoch+1}] Train loss={train_loss:.3f}, Valid RMSE={val_rmse:.4f}")

            # Early stopping check
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Stopped early at epoch {epoch+1}. Best RMSE: {best_rmse:.4f}")
                break
            
        # Load best model back
        self.load_state_dict(best_model_state)


    def predict(self, sids: np.ndarray, pids: np.ndarray):
        """
        Predicts ratings for the given arrays of user and item IDs.

        Input:
            sids (np.ndarray): Array of scientist IDs.
            pids (np.ndarray): Array of paper IDs.

        Outputs:
            np.ndarray: Predicted ratings clipped to [1, 5].
        """
        self.eval()
        self.to(device)
        with torch.no_grad():
            ratings = self.forward(
                torch.from_numpy(sids).to(device),
                torch.from_numpy(pids).to(device)
            )
        return ratings.clamp(1, 5).cpu().numpy()
    

    def export(self):
        """
        Saves the model's learned weights to a file.
        """
        file_path = os.path.join("models", self.name + ".pth")
        torch.save(self.state_dict(), file_path)
        print(f"Model and metadata exported to {file_path}")
    

    @classmethod
    def load(cls, name, mf_dim=64, mlp_layer_sizes=[128,64,32], dropout=0.3):
        """
        Loads a saved NeuMF model from disk.

        Input:
            name (str): Name of the model file (without extension).
            mf_dim (int): GMF embedding dimension (used to reconstruct model).
            mlp_layer_sizes (list): MLP layer sizes.
            dropout (float): Dropout rate used in MLP.

        Output:
            neuMF: Loaded model instance.
        """
        path = os.path.join("models", name + ".pth")

        model = neuMF(
            name=name,
            mf_dim=mf_dim,
            mlp_layer_sizes=mlp_layer_sizes,
            dropout=dropout
        )
        model.load_state_dict(torch.load(path))

        print(f"Loaded NeuMF model and metadata from {path}")
        return model