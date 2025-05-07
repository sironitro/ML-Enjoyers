from als import AlternatingLeastSquares
from bfm import BFM
from svdpp import SVDpp
from neuMF import neuMF
from utils import *


    
if __name__ == '__main__':
    train_df, valid_df, implicit_df = read_data_df(random_state=42)

    # ALS Model Training
    als_model = AlternatingLeastSquares("als", 300, 32)
    als_model.train(train_df)
    print(evaluate(valid_df, als_model.predict))
    als_model.export()
    als_model = AlternatingLeastSquares.load("als")
    print(evaluate(valid_df, als_model.predict))

    # SVDpp Model Training
    scientist2wishlist = read_wishlist_dict()
    scientist2papers = train_df.groupby("sid")["pid"].apply(list).to_dict()
    global_mean = torch.tensor(np.mean(train_df.rating.values), dtype=torch.float32)
    svdpp_model = SVDpp(name="svdpp", epochs=300, s2p=scientist2papers, s2w=scientist2wishlist, global_mean=global_mean)
    svdpp_model.train_model(train_df, valid_df)
    print(evaluate(valid_df, svdpp_model.predict))
    svdpp_model.export()
    svdpp_model = SVDpp.load("svdpp", scientist2papers, scientist2wishlist, global_mean)
    print(evaluate(valid_df, svdpp_model.predict))

    # NCF Model Training
    name="neuMF"
    neuMF_model = neuMF(
        name=name,
        epochs=1
    )
    neuMF_model.train_model(
        train_df,
        valid_df
    )
    print(evaluate(valid_df, neuMF_model.predict))
    neuMF_model.export()
    neuMF_model = neuMF.load(name, 64, [128, 64, 32], dropout=0.3)
    print(evaluate(valid_df, neuMF_model.predict))

    # BFM Models Training
    name="bfm"
    bfm_mpdel = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=None,
        ordinal=False
    )
    bfm_mpdel.training(train_df)
    print(evaluate(valid_df, bfm_mpdel.predict))
    bfm_mpdel.export()
    bfm_mpdel = BFM.load(name)
    print(evaluate(valid_df, bfm_mpdel.predict))


    name="bfm_impl"
    bfm_impl_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=implicit_df,
        ordinal=False
    )
    bfm_impl_model.training(train_df)
    print(evaluate(valid_df, bfm_impl_model.predict))
    bfm_impl_model.export()
    bfm_impl_model = BFM.load(name)
    print(evaluate(valid_df, bfm_impl_model.predict))

    name="bfm_or"
    bfm_or_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=None,
        ordinal=True
    )
    bfm_or_model.training(train_df)
    print(evaluate(valid_df, bfm_or_model.predict))
    bfm_or_model.export()
    bfm_or_model = BFM.load(name)
    print(evaluate(valid_df, bfm_or_model.predict))

    name="bfm_impl_or"
    bfm_impl_or_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=implicit_df,
        ordinal=True
    )
    bfm_impl_or_model.training(train_df)
    print(evaluate(valid_df, bfm_impl_or_model.predict))
    bfm_impl_or_model.export()
    bfm_impl_or_model = BFM.load(name)
    print(evaluate(valid_df, bfm_impl_or_model.predict))





