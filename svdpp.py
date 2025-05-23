from model import Model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import get_dataset
import copy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(SEED)


class SVDpp(Model, nn.Module):
    """
    Implementation of the SVD++ algorithm for collaborative filtering.
    Incorporates both explicit and implicit feedback from scientists and papers, including wishlist behavior. 
    Optionally supports contrastive learning to improve latent representation 
    by pushing apart negative item interactions from positive ones.
    """
    def __init__(self, name: str = "svdpp", epochs = 300, num_scientists: int = 10000, num_papers: int = 1000, emb_dim: int = 64, s2p: dict = dict(), s2w: dict = dict(), global_mean: torch.float32 = 3.82, contrastive_learning: bool = False):
        """
        Initializes model parameters and embeddings for users, items, and implicit feedback.

        Input:
            name (str): Name of the model instance.
            epochs (int): Number of training epochs.
            num_scientists (int): Total number of scientists.
            num_papers (int): Total number of papers.
            emb_dim (int): Dimensionality of latent embeddings.
            s2p (dict): Mapping from scientist ID to papers they rated (implicit feedback).
            s2w (dict): Mapping from scientist ID to papers on their wishlist (additional implicit feedback).
            global_mean (float): Global mean rating used for prediction.
            contrastive_learning (bool): Whether to apply contrastive learning regularization during training.
        """
        nn.Module.__init__(self)
        self.name = name
        self.emb_dim = emb_dim
        self.s2p = s2p
        self.s2w = s2w
        self.epochs = epochs
        self.global_mean = global_mean
        self.contrastive_learning = contrastive_learning # Boolean that activates contrastive learning

        # Embeddings for scientists and papers
        self.scientist_factors = nn.Embedding(num_scientists, emb_dim)
        self.paper_factors = nn.Embedding(num_papers, emb_dim)
        self.scientist_bias = nn.Embedding(num_scientists, 1)
        self.paper_bias = nn.Embedding(num_papers, 1)
        self.implicit_factors = nn.Embedding(num_papers, emb_dim)
        self.implicit_wishlist = nn.Embedding(num_papers, emb_dim)

        # Global average rating
        self.global_bias = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)

        # Init weights
        nn.init.normal_(self.scientist_factors.weight, std=0.1)
        nn.init.normal_(self.paper_factors.weight, std=0.1)
        nn.init.normal_(self.implicit_factors.weight, std=0.1)
        nn.init.normal_(self.implicit_wishlist.weight, std=0.1)
        nn.init.constant_(self.scientist_bias.weight, 0.0)
        nn.init.constant_(self.paper_bias.weight, 0.0)
    

    def forward(self, scientist_ids, paper_ids):
        """
        Computes the predicted ratings for given scientist-paper pairs.

        Input:
            scientist_ids (Tensor): Tensor of scientist IDs.
            paper_ids (Tensor): Tensor of paper IDs.

        Outputs:
            Tensor: Predicted ratings.
        """
        # Latent factors and biases for current batch
        scientist_embeddings = self.scientist_factors(scientist_ids)
        paper_embeddings = self.paper_factors(paper_ids)
        # Squeeze to remove extra dim
        scientist_biases = self.scientist_bias(scientist_ids).squeeze()
        paper_biases = self.paper_bias(paper_ids).squeeze()

        papers = [self.s2p.get(k, []) for k in scientist_ids]

        # Compute Implicit feedback from rated papers
        implicit_embeds = []
        for sp in papers:
            if len(sp) > 0:
                y_j = self.implicit_factors(torch.tensor(sp, device=device))
                sum_yj = y_j.sum(dim=0)
                norm_yj = sum_yj / torch.sqrt(torch.tensor(len(sp), dtype=torch.float, device=device))
            else:
                norm_yj = torch.zeros_like(scientist_embeddings[0])
            implicit_embeds.append(norm_yj)
        y_u = torch.stack(implicit_embeds)


        # Compute Implicit feedback from wishlist papers
        wishlist = [self.s2w.get(k, []) for k in scientist_ids]

        implicit_embeds_wl = []
        for w in wishlist:
            if len(w) > 0:
                y_j_wl = self.implicit_wishlist(torch.tensor(w, device=device))
                sum_yj_wl = y_j_wl.sum(dim=0)
                norm_yj_wl = sum_yj_wl / torch.sqrt(torch.tensor(len(w), dtype=torch.float, device=device))
            else:
                norm_yj_wl = torch.zeros_like(scientist_embeddings[0])
            implicit_embeds_wl.append(norm_yj_wl)
        y_u_wl = torch.stack(implicit_embeds_wl)

        # Combine explicit and implicit interactions
        interaction = ((scientist_embeddings + y_u + y_u_wl)  * paper_embeddings).sum(dim=1)

        # Predict ratings
        predicted_ratings = interaction + scientist_biases + paper_biases + self.global_bias

        return predicted_ratings

        
    def train_model(self, train_df, valid_df):
        """
        Trains the SVD++ model using training and validation datasets.
        Uses early stopping based on RMSE on validation data.

        Input:
            train_df (DataFrame): Training data (sid, pid, rating).
            valid_df (DataFrame): Validation data (sid, pid, rating).
        """
        # Convert dataframes to PyTorch datasets and setup data loaders
        train_dataset = get_dataset(train_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataset = get_dataset(valid_df)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
        
        best_rmse = float("inf")
        patience = 2
        epochs_no_improve = 0
        best_model_state = None
        model = self.to(device)
        
        # Set up Adam optimizer
        optim = torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=4e-5)

        # Training loop
        for epoch in range(self.epochs):
            # Train model for an epoch
            total_loss = 0.0
            total_data = 0
            model.train()
            # Loop over training batches
            for sid, pid, ratings in train_loader:
                sid = sid.to(device)
                pid = pid.to(device)
                ratings = ratings.to(device)

                pred = model(sid, pid)
                loss = F.mse_loss(pred, ratings)

                if self.contrastive_learning:
                    # Contrastive loss
                    neg_pids = torch.randint(0, 1000, pid.shape, device=device, generator=generator)

                    user_embeds = model.scientist_factors(sid)
                    pos_item_embeds = model.paper_factors(pid)
                    neg_item_embeds = model.paper_factors(neg_pids)

                    pos_scores = F.cosine_similarity(user_embeds, pos_item_embeds)
                    neg_scores = F.cosine_similarity(user_embeds, neg_item_embeds)

                    contrast_loss = self.contrastive_loss(pos_scores, neg_scores)
                    loss += 0.2 * contrast_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_data += len(sid)
                total_loss += len(sid) * loss.item()

            # Evaluate on validation set
            total_val_mse = 0.0
            total_val_data = 0
            model.eval()
            for sid, pid, ratings in valid_loader:
                sid = sid.to(device)
                pid = pid.to(device)
                ratings = ratings.to(device)

                pred = model(sid, pid).clamp(1, 5)
                mse = F.mse_loss(pred, ratings)

                total_val_data += len(sid)
                total_val_mse += len(sid) * mse.item()

            val_rmse = (total_val_mse / total_val_data) ** 0.5
            train_loss = total_loss / total_data
            print(f"[Epoch {epoch+1}] Train loss={train_loss:.3f}, Valid RMSE={val_rmse:.4f}")

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

        # Load best model back
        model.load_state_dict(best_model_state)


    def export(self):
        """
        Saves the model's learned parameters to a file.
        """
        file_path = os.path.join("models", self.name + ".pth")
        torch.save(self.state_dict(), file_path)
        print(f"Model and metadata exported to {file_path}")
        
    
    def predict(self, sids: np.ndarray, pids: np.ndarray):
        """
        Makes predictions for the given arrays of scientist and paper IDs.

        Input:
            sids (np.ndarray): Array of scientist IDs.
            pids (np.ndarray): Array of paper IDs.

        Outputs:
            np.ndarray: Array of predicted ratings.
        """
        self.to(device)
        with torch.no_grad():
            return self.forward(
                torch.from_numpy(sids).to(device),
                torch.from_numpy(pids).to(device),
            ).clamp(1, 5).cpu().numpy()
    

    @classmethod
    def load(cls, name, s2p, s2w, global_mean, contrastive_learning = False):
        """
        Loads a saved SVDpp model from disk.

        Input:
            name (str): Name of the model file (without extension).
            s2p (dict): Mapping from scientist ID to papers they rated (implicit feedback).
            s2w (dict): Mapping from scientist ID to papers on their wishlist (additional implicit feedback).
            global_mean (float): Global mean rating to reinitialize the model.
            contrastive_learning (bool): Whether to apply contrastive learning regularization during training.

        Outputs:
            SVDpp: An instance of the SVDpp model with loaded weights.
        """
        path = os.path.join("models", name + ".pth")

        model = SVDpp(s2p=s2p, s2w=s2w, global_mean=global_mean, contrastive_learning=contrastive_learning)
        model.load_state_dict(torch.load(path))

        print(f"Loaded SVDpp model and metadata from {path}")
        return model
    

    def contrastive_loss(self, pos_scores, neg_scores, margin=1.0):
        """
        Computes a contrastive hinge loss to encourage positive item scores to be higher than negative ones.

        Inputs:
            pos_scores (Tensor): Cosine similarity scores between user and positively interacted item embeddings.
            neg_scores (Tensor): Cosine similarity scores between user and negatively sampled item embeddings.
            margin (float): Desired minimum margin between positive and negative scores (default: 1.0).

        Outputs:
            Tensor: Scalar contrastive loss value.
        """
        # Want pos_scores >> neg_scores, so hinge loss on margin
        loss = F.relu(margin - pos_scores + neg_scores)
        return loss.mean()
