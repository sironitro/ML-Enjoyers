"""
Model Training Script

Trains, evaluates, exports, and reloads  models used in the ensemble:
ALS, SVD++, NCF, and BFM (with various configurations).

Each model is trained on the same dataset and evaluated using RMSE on a validation set.
Models are saved after training and reloaded to verify consistent performance when exporting and reloading.
"""
from als import AlternatingLeastSquares
from bfm import BFM
from svdpp import SVDpp
from neuMF import neuMF
from utils import *


    
if __name__ == '__main__':
    # Get training, validation and implicit dataframe from files
    train_df, valid_df, implicit_df = read_data_df(random_state=42)

    # ALS Model Training
    # Initialize and train the ALS model
    als_model = AlternatingLeastSquares("als", 300, 32)
    als_model.train(train_df)
    # Evaluate on validation set
    print(evaluate(valid_df, als_model.predict))
    # Export the trained model to disk
    als_model.export()
    # Load the model back from disk and evaluate
    als_model = AlternatingLeastSquares.load("als")
    print(evaluate(valid_df, als_model.predict))

    # SVDpp Model Training
    # Prepare implicit feedback data and global mean
    scientist2wishlist = read_wishlist_dict()
    scientist2papers = train_df.groupby("sid")["pid"].apply(list).to_dict()
    global_mean = torch.tensor(np.mean(train_df.rating.values), dtype=torch.float32)
    # Initialize and train the SVD++ model
    svdpp_model = SVDpp(name="svdpp", epochs=300, s2p=scientist2papers, s2w=scientist2wishlist, global_mean=global_mean)
    svdpp_model.train_model(train_df, valid_df)
    # Evaluate on validation set
    print(evaluate(valid_df, svdpp_model.predict))
    # Export the trained model to disk
    svdpp_model.export()
    # Load the model back from disk and evaluate
    svdpp_model = SVDpp.load("svdpp", scientist2papers, scientist2wishlist, global_mean)
    print(evaluate(valid_df, svdpp_model.predict))

    # NCF Model Training
    # Initialize and train the NCF model
    name="neuMF"
    neuMF_model = neuMF(
        name=name,
        epochs=1
    )
    neuMF_model.train_model(
        train_df,
        valid_df
    )
    # Evaluate on validation set
    print(evaluate(valid_df, neuMF_model.predict))
    # Export the trained model to disk
    neuMF_model.export()
    # Load the model back from disk and evaluate
    neuMF_model = neuMF.load(name, 64, [128, 64, 32], dropout=0.3)
    print(evaluate(valid_df, neuMF_model.predict))

    # BFM Models Training
    # Initialize and train the standard BFM model
    name="bfm"
    bfm_mpdel = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=None,
        ordinal=False
    )
    bfm_mpdel.training(train_df)
    # Evaluate on validation set
    print(evaluate(valid_df, bfm_mpdel.predict))
    # Export the trained model to disk
    bfm_mpdel.export()
    # Load the model back from disk and evaluate
    bfm_mpdel = BFM.load(name)
    print(evaluate(valid_df, bfm_mpdel.predict))

    # Initialize and train the BFM model with implicit features
    name="bfm_impl"
    bfm_impl_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=implicit_df,
        ordinal=False
    )
    bfm_impl_model.training(train_df)
    # Evaluate on validation set
    print(evaluate(valid_df, bfm_impl_model.predict))
    # Export the trained model to disk
    bfm_impl_model.export()
    # Load the model back from disk and evaluate
    bfm_impl_model = BFM.load(name)
    print(evaluate(valid_df, bfm_impl_model.predict))

    # Initialize and train the BFM model with ordinal regression
    name="bfm_or"
    bfm_or_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=None,
        ordinal=True
    )
    bfm_or_model.training(train_df)
    # Evaluate on validation set
    print(evaluate(valid_df, bfm_or_model.predict))
    # Export the trained model to disk
    bfm_or_model.export()
    # Load the model back from disk and evaluate
    bfm_or_model = BFM.load(name)
    print(evaluate(valid_df, bfm_or_model.predict))

    # Initialize and train the BFM model with imiplicit features and ordinal regression
    name="bfm_impl_or"
    bfm_impl_or_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=implicit_df,
        ordinal=True
    )
    bfm_impl_or_model.training(train_df)
    # Evaluate on validation set
    print(evaluate(valid_df, bfm_impl_or_model.predict))
    # Export the trained model to disk
    bfm_impl_or_model.export()
    # Load the model back from disk and evaluate
    bfm_impl_or_model = BFM.load(name)
    print(evaluate(valid_df, bfm_impl_or_model.predict))





