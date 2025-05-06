from als import AlternatingLeastSquares
from bfm import BFM
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
    
    name="bfm_or"
    bfm_model = BFM(
        name=name,
        n_iter=300,
        rank=32,
        implicit_df=None,
        ordinal=True
    )
    bfm_model.train(train_df)

        
    
    print(evaluate(valid_df, bfm_model.predict))
    bfm_model.export()
    loaded_model = BFM.load(name)
    print(evaluate(valid_df, loaded_model.predict))