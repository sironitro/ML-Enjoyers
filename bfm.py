import os
from model import Model
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import myfm
from myfm import RelationBlock
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse as sps



def return_zero():
    return 0


class BFM(Model):
    """
    Bayesian Factorization Machine (BFM) using the myFM library.
    Supports optional use of implicit features or orindal regression.
    """
    def __init__(self, name: str, n_iter, rank, implicit_df : pd.DataFrame, ordinal : bool):
        """
        Initializes the BFM model.

        Input:
            name (str): Name of the model instance.
            n_iter (int): Number of training iterations.
            rank (int): Latent factor dimension.
            implicit_df (DataFrame): Implicit feedback data (sid, pid) or None.
            ordinal (bool): Whether to use ordinal probit model or regression.
        """
        self.name = name
        self.implicit_df = implicit_df
        self.ordinal = ordinal
        self.n_iter = n_iter
        if ordinal:
            self.model = myfm.MyFMOrderedProbit(rank=rank, random_seed=42)
        else:
            self.model = myfm.MyFMRegressor(rank=rank, random_seed=42)
        self.ohe = OneHotEncoder(handle_unknown='ignore')
    

    def augment_scientist_id(self, scientist_ids):
        """
        Constructs augmented scientist feature matrix with implicit interactions.

        Input:
            scientist_ids (array-like): List of scientist IDs
        
        Outputs:
            csr_matrix: Augmented scientist feature matrix with optional implicit info
        """
        Xs = []

        # One-hot encode scientist ID
        X_sid = sps.lil_matrix((len(scientist_ids), self.SCIENTIST_ID_SIZE))
        for index, scientist_id in enumerate(scientist_ids):
            X_sid[index, self.scientist_to_index[scientist_id]] = 1
        Xs.append(X_sid)

         # Add implicit feedback: papers the scientist has interacted with
        X_is = sps.lil_matrix((len(scientist_ids), self.PAPER_ID_SIZE))
        for index, scientist_id in enumerate(scientist_ids):
            read_papers = self.scientist_vs_read.get(scientist_id, [])
            normalizer = 1 / max(len(read_papers), 1) ** 0.5
            for sid in read_papers:
                X_is[index, self.paper_to_index[sid]] = normalizer
        Xs.append(X_is)

        return sps.hstack(Xs, format='csr')
            
            
    def augment_paper_id(self, paper_ids):
        """
        Constructs augmented paper feature matrix with implicit interactions.

        Input:
            paper_ids (array-like): List of paper IDs
        
        Outputs:
            csr_matrix: Augmented paper feature matrix with optional implicit info
        """
        Xs = []

        # One-hot encode paper ID
        X_paper = sps.lil_matrix((len(paper_ids), self.PAPER_ID_SIZE))
        for index, paper_id in enumerate(paper_ids):
            X_paper[index, self.paper_to_index[paper_id]] = 1
        Xs.append(X_paper)

        # Add implicit feedback: scientists who interacted with this paper
        X_ip = sps.lil_matrix((len(paper_ids), self.SCIENTIST_ID_SIZE))
        for index, paper_id in enumerate(paper_ids):
            read_scientists = self.paper_vs_read.get(paper_id, [])
            normalizer = 1 / max(len(read_scientists), 1) ** 0.5
            for sid in read_scientists:
                X_ip[index, self.scientist_to_index[sid]] = normalizer
        Xs.append(X_ip)

        return sps.hstack(Xs, format='csr')
    

    def construct_relation_blocks(self, df):    
        """
        Builds relation blocks from the given DataFrame based on implicit features.

        Input:
            df (DataFrame): Data containing "sid" and "pid"

        Outputs:
            Tuple[RelationBlock, RelationBlock]: relation blocks for scientists and papers
        """
        sid_unique, sid_index = np.unique(df.sid, return_inverse=True)
        pid_unique, pid_index = np.unique(df.pid, return_inverse=True)

         # Generate implicit feature matrices
        scientist_data = self.augment_scientist_id(sid_unique)
        paper_data = self.augment_paper_id(pid_unique)

        # Wrap into myFM relation blocks        
        block_scientist = RelationBlock(sid_index, scientist_data)
        block_paper = RelationBlock(pid_index, paper_data)

        return block_scientist, block_paper


    def train_model(self, train_df):
        """
        Trains the BFM model using the provided training data and optional implicit feedback.

        Input:
            train_df (DataFrame): Training data with columns "sid", "pid", "rating"
        """
        # One-hot encode (sid, pid)
        X_train = self.ohe.fit_transform(train_df[['sid', 'pid']])
        y_train = train_df.rating.values

        self.X_rel = []
        self.group_shapes = None
        if self.ordinal:
            group_shapes = [len(group) for group in self.ohe.categories_]

        # If implicit data exists, construct relation blocks and update X_rel (and group_shapes if needed)
        if self.implicit_df is not None:
            # Build ID mappings
            self.scientist_to_index = defaultdict(return_zero, { sid: i+1 for i,sid in enumerate(np.unique(train_df.sid))})
            self.paper_to_index = defaultdict(return_zero, { pid: i+1 for i,pid in enumerate(np.unique(train_df.pid))})
            self.SCIENTIST_ID_SIZE = len(self.scientist_to_index) + 1
            self.PAPER_ID_SIZE = len(self.paper_to_index) + 1
        
            # Build interaction history
            self.paper_vs_read = dict()
            self.scientist_vs_read = dict()
            for row in train_df.itertuples():
                sid = row.sid
                pid = row.pid
                self.paper_vs_read.setdefault(pid, list()).append(sid)
                self.scientist_vs_read.setdefault(sid, list()).append(pid)
            
            for row in self.implicit_df.itertuples():
                sid = row.sid
                pid = row.pid
                self.paper_vs_read.setdefault(pid, list()).append(sid)
                self.scientist_vs_read.setdefault(sid, list()).append(pid)
            
            # Build relation blocks
            block_scientist_train, block_paper_train = self.construct_relation_blocks(train_df)
            self.X_rel = [block_scientist_train, block_paper_train]

            if self.group_shapes is not None:
                self.group_shapes.extend([block_scientist_train.feature_size, block_paper_train.feature_size])
        
        self.model.fit(
            X_train, 
            y_train-1 if self.ordinal else y_train, 
            X_rel=self.X_rel,
            group_shapes=self.group_shapes,
            n_iter=self.n_iter, n_kept_samples=self.n_iter
        )


    def export(self):
        """
        Saves the trained BFM model and associated data to disk.
        """
        with open(f'models/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        
    
    def predict(self, sids: np.ndarray, pids: np.ndarray):
        """
        Predicts ratings for the given (sid, pid) pairs.

        Input:
            sids (np.ndarray): Array of scientist IDs.
            pids (np.ndarray): Array of paper IDs.

        Outputs:
            np.ndarray: Predicted ratings (clipped to range [1, 5]).
        """
        pred_df = pd.DataFrame({'sid': sids, 'pid': pids})
        X = self.ohe.transform(pred_df[['sid', 'pid']])
        X_rel = []
        
        # Rebuild relation blocks if model was trained with implicit features
        if self.implicit_df is not None:            
            block_scientist_pred, block_paper_pred = self.construct_relation_blocks(pred_df)
            X_rel=[block_scientist_pred, block_paper_pred]
        
        # Compute prediction with oridinal regression if ordinal is set to True
        if self.ordinal:
            p_ordinal = self.model.predict_proba(
                X, 
                X_rel=X_rel
            )
            expected_rating = p_ordinal.dot(np.arange(1, 6))
            expected_rating = expected_rating.clip(1,5) 
            return expected_rating  
        else:
            # Predict usual regression scores
            return self.model.predict(
                X,
                X_rel = X_rel
            ).clip(1,5) 
            
            
    @classmethod
    def load(cls, name):
        """
        Loads a previously saved BFM model from disk.

        Input:
            name (str): Model filename without extension.

        Outputs:
            BFM: A restored model instance.
        """
        path = os.path.join("models", name + ".pkl")

        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"Loaded BFM model and metadata from {path}")
        return obj