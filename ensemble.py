from als import AlternatingLeastSquares
from bfm import BFM
from svdpp import SVDpp
from neuMF import neuMF
from utils import *

# class Ensemble():
#     self.models
    
    
if __name__ == '__main__':
    train_df, valid_df, implicit_df = read_data_df(random_state=42)
    # als_model = AlternatingLeastSquares("als", 300, 32)
    # als_model.train(train_df)
    # print(evaluate(valid_df, als_model.predict))
    # als_model.export()
    
    
    
    # loaded_model = AlternatingLeastSquares.load("als")
    # print(evaluate(valid_df, loaded_model.predict))
    
    # name="bfm_or"
    # bfm_model = BFM(
    #     name=name,
    #     n_iter=300,
    #     rank=32,
    #     implicit_df=None,
    #     ordinal=True
    # )
    # bfm_model.training(train_df)
    
        
    
    # print(evaluate(valid_df, bfm_model.predict))
    # bfm_model.export()
    # loaded_model = BFM.load(name)
    # print(evaluate(valid_df, loaded_model.predict))
    
    # scientist2wishlist = read_wishlist_dict()
    # scientist2papers = train_df.groupby("sid")["pid"].apply(list).to_dict()
    # global_mean = torch.tensor(np.mean(train_df.rating.values), dtype=torch.float32)
    # svdpp_model = SVDpp(name="svdpp", epochs=1, s2p=scientist2papers, s2w=scientist2wishlist, global_mean=global_mean)
    # svdpp_model.train_model(train_df, valid_df)
    # print(evaluate(valid_df, svdpp_model.predict))
    # svdpp_model.export()
    
    # loaded_model = SVDpp.load("svdpp", scientist2papers, scientist2wishlist, global_mean)
    # print(evaluate(valid_df, loaded_model.predict))
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
    loaded_model = neuMF.load(name, 64, [128, 64, 32], dropout=0.3)
    print(evaluate(valid_df, loaded_model.predict))
