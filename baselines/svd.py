import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def impute_values(mat: np.ndarray) -> np.ndarray:
    """
    Replaces all NaN values in the input matrix with the default value 3.0.

    Input:
        mat (np.ndarray): Input matrix possibly containing NaNs.

    Outputs:
        np.ndarray: Matrix with NaNs replaced by 3.0.
    """
    return np.nan_to_num(mat, nan=3.0)


def opt_rank_k_approximation(m: np.ndarray, k: int):
    """
    Returns the optimal rank-k reconstruction matrix, using SVD.

    Input:
        m (np.ndarray): Input matrix to be approximated.
        k (int): Target rank for the approximation. Must satisfy 0 < k <= min(m.shape).

    Outputs:
        np.ndarray: Reconstructed matrix of rank k.
    """
    assert 0 < k <= np.min(m.shape), f"The rank must be in [0, min(m, n)]"
    
    U, S, Vh = np.linalg.svd(m, full_matrices=False)
    
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k]
    
    return np.dot(U_k * S_k, Vh_k)


def matrix_pred_fn(train_recon: np.ndarray, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Looks up predicted ratings from a precomputed matrix for given (sid, pid) pairs.

    Input:
        train_recon (np.ndarray): Matrix of predicted ratings (shape: [num_scientists, num_papers]).
        sids (np.ndarray): Array of scientist (user) IDs.
        pids (np.ndarray): Array of paper (item) IDs.
        
    Outputs:
        np.ndarray: Array of predicted ratings corresponding to (sid, pid) pairs.
    """
    return train_recon[sids, pids]



# Get train and validation data and setup data loaders
train_df, valid_df, _ = read_data_df()
train_mat = read_data_matrix(train_df)
train_mat = impute_values(train_mat)

# Compute and plot singular values
singular_values = np.linalg.svd(train_mat, compute_uv=False, hermitian=False)
plt.plot(singular_values)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Singular value spectrum")
plt.show()


# Compute the rank-k approximation of the training matrix
train_recon = opt_rank_k_approximation(train_mat, k=2)

# Define prediction function
pred_fn = lambda sids, pids: matrix_pred_fn(train_recon, sids, pids)

# Evaluate on validation data and get submission
val_score = evaluate(valid_df, pred_fn)
print(f"Validation RMSE: {val_score:.3f}")
make_submission(pred_fn, "submissions/svd_submission.csv")