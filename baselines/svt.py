import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils import *
from numpy.linalg import svd



def svt(M_obs, Omega, tau=500, delta=1.2, max_iter=20, tol=1e-4):
    """
    M_obs : np.ndarray
        Matrix with missing entries filled with 0 (or any constant), but only used with Omega.
    Omega : np.ndarray
        Binary mask (1 where data is observed, 0 where missing).
    tau : float
        Thresholding parameter for singular values.
    delta : float
        Step size for updating Y.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold.
    """
    m, n = M_obs.shape
    Y = np.zeros((m, n))  # Initialize dual variable

    for k in range(max_iter):
        print(f"Iteration {k}")

        # Step 1: SVD
        try:
            U, S, Vt = svd(Y, full_matrices=False)
        except np.linalg.LinAlgError:
            print("SVD failed to converge. Consider reducing delta or tau.")
            break

        # Step 2: Soft-thresholding
        S_thresh = np.maximum(S - tau, 0)
        X = U @ np.diag(S_thresh) @ Vt

        # Step 3: Project X onto Omega (keep only observed entries)
        X_Omega = Omega * X

        # Step 4: Update Y using only observed values
        residual = M_obs - X_Omega
        Y += delta * residual

        # Step 5: Convergence check (on observed entries only)
        norm_diff = np.linalg.norm(residual, 'fro')
        norm_orig = np.linalg.norm(Omega * M_obs, 'fro')
        err = norm_diff / (norm_orig + 1e-8)

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

completed_mat = svt(train_mat, omega)
completed_mat += mu
completed_mat *= sigma
completed_mat = np.clip(completed_mat, 1, 5)

# Prediction function
def svt_predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    preds = []
    for sid, pid in zip(sids, pids):
        if sid < completed_mat.shape[0] and pid < completed_mat.shape[1]:
            preds.append(completed_mat[sid, pid])
        else:
            preds.append(np.nanmean(completed_mat))
    return np.array(preds)

# Evaluate
rmse = evaluate(valid_df, svt_predict)
print(f"Validation RMSE (Custom SVT): {rmse:.4f}")

# Make Submission
make_submission(svt_predict, "submissions/svt_submission.csv")