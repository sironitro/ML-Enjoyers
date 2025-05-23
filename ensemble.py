from als import AlternatingLeastSquares
from bfm import BFM
from svdpp import SVDpp
from neuMF import neuMF
from utils import *
from sklearn.linear_model import Ridge, LinearRegression



class Ensemble():
    """
    Ensemble model that combines predictions from multiple base recommendation models
    using a Ridge regression meta-model. The ensemble is trained on validation data
    and used to make final predictions.

    Models included:
        - SVD++ (with and without Contrastive Learning)
        - Alternating Least Squares (ALS)
        - Bayesian Factorization Machines (BFM, with multiple variants)
        - Neural Collaborative Filtering (NCF)
    """
    def __init__(self):
        """
        Initializes the ensemble by loading data and models.
        """
        self.train_df, self.valid_df, self.implicit_df = read_data_df(random_state=42)
        self.scientist2wishlist = read_wishlist_dict()
        self.scientist2papers = self.train_df.groupby("sid")["pid"].apply(list).to_dict()
        self.global_mean = torch.tensor(np.mean(self.train_df.rating.values), dtype=torch.float32)
        # Load pre-trained base models
        self.models = [
            SVDpp.load("svdpp", self.scientist2papers, self.scientist2wishlist, self.global_mean),
            SVDpp.load("svdpp_cl", self.scientist2papers, self.scientist2wishlist, self.global_mean, True),
            AlternatingLeastSquares.load("als"),
            BFM.load("bfm"),
            BFM.load("bfm_impl"),
            BFM.load("bfm_impl_or"),
            BFM.load("bfm_or"),
            neuMF.load("neuMF")
        ]
        self.meta_model = Ridge(alpha=1.0)


    def get_model_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Collects predictions from all base models on the validation set.

        Outputs:
            Tuple[np.ndarray, np.ndarray]: A matrix of shape (n_samples, n_models)
            with predictions from each model, and the true ratings vector.
        """
        print("getting model predictions ...")
        sids = self.valid_df['sid'].to_numpy()
        pids = self.valid_df['pid'].to_numpy()
        preds = [model.predict(sids, pids) for model in self.models]
        return np.stack(preds, axis=1)
    
    
    def fit(self):
        """
        Fits the Ridge regression meta-model on the validation predictions.
        """
        X_val = self.get_model_predictions()
        y_val = self.valid_df['rating']
        print("fitting meta model ...")
        self.meta_model.fit(X_val, y_val)
    
    
    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """
        Predicts ratings using the ensemble model.

        Input:
            sids (np.ndarray): Scientist IDs
            pids (np.ndarray): Paper IDs

        Outputs:
            np.ndarray: Predicted ratings
        """
        print("predicting ...")
        preds = [model.predict(sids, pids) for model in self.models]
        X = np.stack(preds, axis=1)
        return self.meta_model.predict(X)
    
    
    def get_weights(self):
        """
        Returns the learned weights for each base model.
        
        Outputs:
            Tuple[np.ndarray, float]: Model coefficients and intercept
        """
        return self.meta_model.coef_, self.meta_model.intercept_
    
    
    def get_individual_submissions(self):
        """
        Generates and saves Kaggle-style submission files for each base model.
        """
        for model in self.models:
            make_submission(model.predict, f"submissions/{model.name}_submission.csv")
    

    
if __name__ == '__main__':
    # Create ensemble, train it and create final submission
    ensemble = Ensemble()
    ensemble.fit()
    make_submission(ensemble.predict, 'submissions/ensemble_submission.csv')
    