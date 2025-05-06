import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.linalg import svd
from utils import *


def svp(M_obs, Omega, delta=1.4, max_iter=300, tol=1e-4, rank=32):
    """
    Singular Value Projection (SVP) for Matrix Completion.
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



# Load and prepare data
train_df, valid_df, _ = read_data_df()
train_mat = read_data_matrix(train_df)
mu = np.nanmean(train_mat, axis=0)
sigma = np.nanstd(train_mat, axis=0)
train_mat = (train_mat - mu) / sigma
omega = ~np.isnan(train_mat)
train_mat = np.nan_to_num(train_mat, nan=0.0)

completed_mat = svp(train_mat, omega)
completed_mat += mu
completed_mat *= sigma
completed_mat = np.clip(completed_mat, 1, 5)

# Prediction function
def svp_predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    preds = []
    for sid, pid in zip(sids, pids):
        if sid < completed_mat.shape[0] and pid < completed_mat.shape[1]:
            preds.append(completed_mat[sid, pid])
        else:
            preds.append(np.nanmean(completed_mat))
    return np.array(preds)


# Evaluate
rmse = evaluate(valid_df, svp_predict)
print(f"Validation RMSE (Custom SVP): {rmse:.4f}")

# Make Submission
make_submission(svp_predict, "submissions/svp_submission.csv")