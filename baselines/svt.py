import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils import *
from numpy.linalg import svd



def svt(M_obs, Omega, tau=500, delta=1.2, max_iter=20, tol=1e-4):
    """
    Singular Value Thresholding (SVT) algorithm for matrix completion.

    Input:
        M_obs (np.ndarray): Matrix with missing entries filled with zeros (only used with Omega).
        Omega (np.ndarray): Binary mask (1 where data is observed, 0 where missing).
        tau (float): Thresholding parameter for singular values (higher = lower rank).
        delta (float): Step size for updating the dual variable Y.
        max_iter (int): Maximum number of iterations.
        tol (float): Relative tolerance for convergence on observed entries.

    Outputs:
        np.ndarray: The completed matrix estimate X.
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


# Prediction function
def svt_predict(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
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

# Perform matrix completion using SVT
completed_mat = svt(train_mat, omega)
completed_mat += mu
completed_mat *= sigma
completed_mat = np.clip(completed_mat, 1, 5)

# Evaluate on validation data and get submission
rmse = evaluate(valid_df, svt_predict)
print(f"Validation RMSE (Custom SVT): {rmse:.4f}")
make_submission(svt_predict, "submissions/svt_submission.csv")