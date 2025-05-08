import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.linalg import svd
from utils import *



def svp(M_obs, Omega, delta=1.4, max_iter=300, tol=1e-4, rank=32):
    """
    Singular Value Projection (SVP) for Matrix Completion.

    Input:
        M_obs (np.ndarray): The observed matrix with missing entries (filled with zeros).
        Omega (np.ndarray): A binary mask of the same shape as M_obs, where 1 indicates observed entries.
        delta (float): Step size for gradient update. Too large may cause divergence.
        max_iter (int): Maximum number of iterations to perform.
        tol (float): Relative error tolerance for convergence.
        rank (int): Target rank for the low-rank approximation.

    Outputs:
        np.ndarray: The completed matrix approximation of M_obs.
    """
    m, n = M_obs.shape
    X = np.zeros((m, n))  # Initial guess

    for k in range(max_iter):
        print(f"Iteration {k}")

        # Gradient step only on observed entries
        G = X + delta * (Omega * (M_obs - X))

        # Project updated guess to rank-r
        try:
            U, S, Vt = svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            print("SVD did not converge. Try reducing delta.")
            break

        U_k = U[:, :rank]
        S_k = np.diag(S[:rank])
        Vt_k = Vt[:rank, :]
        X = U_k @ S_k @ Vt_k

        # Convergence check (on observed entries only)
        residual = Omega * (M_obs - X)
        norm_diff = np.linalg.norm(residual, 'fro')
        norm_obs = np.linalg.norm(Omega * M_obs, 'fro') + 1e-8
        err = norm_diff / norm_obs

        print(f"  Error: {err:.2e}")
        if err < tol:
            print(f"Converged at iteration {k}, error: {err:.2e}")
            break

    return X


# Define prediction function
def svp_predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Predict ratings for given (scientist_id, paper_id) pairs using the completed matrix.

    Input:
        sids (np.ndarray): Array of scientist (user) IDs.
        pids (np.ndarray): Array of paper (item) IDs.

    Outputs:
        np.ndarray: Array of predicted ratings.
    """
    preds = []
    for sid, pid in zip(sids, pids):
        if sid < completed_mat.shape[0] and pid < completed_mat.shape[1]:
            preds.append(completed_mat[sid, pid])
        else:
            preds.append(np.nanmean(completed_mat))
    return np.array(preds)


# Get train and validation data
train_df, valid_df, _ = read_data_df()
train_mat = read_data_matrix(train_df)

# Compute column-wise mean and standard deviation
mu = np.nanmean(train_mat, axis=0)
sigma = np.nanstd(train_mat, axis=0)

# Normalize training matrix, setup mask for observed entries and replace NaNs with zeros
train_mat = (train_mat - mu) / sigma
omega = ~np.isnan(train_mat)
train_mat = np.nan_to_num(train_mat, nan=0.0)

# Perform matrix completion using SVP
completed_mat = svp(train_mat, omega)
completed_mat += mu
completed_mat *= sigma
completed_mat = np.clip(completed_mat, 1, 5)


# Evaluate on validation data and get submission
rmse = evaluate(valid_df, svp_predict)
print(f"Validation RMSE (Custom SVP): {rmse:.4f}")
make_submission(svp_predict, "submissions/svp_submission.csv")