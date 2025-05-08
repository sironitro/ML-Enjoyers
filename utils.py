from typing import Tuple, Callable
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error



DATA_DIR = "./data"


def read_data_df(random_state=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and splits it into training and validation sets with a 75/25 split.

    Input:
        random_state (int or None): Seed for reproducibility.

    Outputs:
        Tuple:
            - train_df: DataFrame with training ratings.
            - valid_df: DataFrame with validation ratings.
            - implicit_df: DataFrame with implicit wishlist interactions.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"))
    implicit_df = pd.read_csv(os.path.join(DATA_DIR, "train_tbr.csv"))

    # Split sid_pid into sid and pid columns
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    
    # Split into train and validation dataset
    train_df, valid_df = train_test_split(df, test_size=0.25, random_state=random_state)
    return train_df, valid_df, implicit_df


def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid).

    Input:
        df (pd.DataFrame): DataFrame containing columns "sid", "pid", and "rating".

    Outputs:
        np.ndarray: A matrix with shape (num_sids, num_pids) where each entry [i, j]
                represents the rating by scientist i for paper j. Missing entries are NaN.
    """

    return df.pivot(index="sid", columns="pid", values="rating").values


def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Evaluates a prediction function on a validation DataFrame using Root Mean Squared Error (RMSE).

    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: 
        float: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)


def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):
    """Makes a submission CSV file that can be submitted to kaggle.

    Inputs:
        pred_fn: Function that takes in arrays of sid and pid and outputs a score.
        filename: File to save the submission to.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values
    
    df["rating"] = pred_fn(sids, pids)
    df.to_csv(filename, index=False)
    
    
def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """
    Converts a pandas DataFrame into a PyTorch TensorDataset.

    Input:
        df (pd.DataFrame): DataFrame containing user IDs ("sid"), item IDs ("pid"),
                       and ratings ("rating").

    Outputs:
        torch.utils.data.Dataset: A TensorDataset where each element is (sid, pid, rating).
    """
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)


def read_wishlist_dict() -> dict:
    """
    Reads the wishlist CSV file and returns a mapping from scientist IDs to
    a list of paper IDs in their wishlist.

    Outputs:
        dict: Dictionary mapping each "sid" to a list of "pid" entries (wishlist papers).
    """
    wishlist = pd.read_csv(os.path.join(DATA_DIR, "train_tbr.csv"))
    wishlist["sid"] = wishlist["sid"].astype(int)
    wishlist["pid"] = wishlist["pid"].astype(int)
    scientist2wishlist = wishlist.groupby("sid")["pid"].apply(list).to_dict()
    return scientist2wishlist