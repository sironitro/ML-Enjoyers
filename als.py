import os
import pickle
from model import Model
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel



class AlternatingLeastSquares(Model):
    """
    PySpark-based implementation of the ALS (Alternating Least Squares) algorithm
    for collaborative filtering.
    """
    def __init__(self, name: str, maxIter, rank):
        """
        Initializes the ALS model and Spark session.
        """
        self._init_spark()
        self.name = name
        self.model = ALS(
            maxIter=maxIter,
            rank=rank,
            userCol="sid",
            itemCol="pid",
            ratingCol="rating",
            seed=42
        )
        
        
    def _init_spark(self):
        """
        Initializes the Spark session used for training and prediction.
        """
        self.spark = SparkSession.builder.appName("ALS") \
            .config("spark.driver.extraJavaOptions", "-Xss16m") \
            .config("spark.executor.extraJavaOptions", "-Xss16m") \
            .getOrCreate()

        
    def train_model(self, train_df):
        """
        Trains the ALS model on the provided training data.
        Skips training and loads model if a saved model already exists.

        Input:
            train_df (pd.DataFrame): DataFrame with columns "sid", "pid", "rating"
        """
        if os.path.exists(f"models/{self.name}") and os.path.isdir(f"models/{self.name}"):
            print(f"model has already been exported models/{self.name}")
            print("Skipping trainig ...")
            self.load()
            return
        s_train_df = self.spark.createDataFrame(train_df)
        self.model = self.model.fit(s_train_df)


    def export(self):
        """
        Saves the trained ALS model and Python object metadata to disk.
        """
        file_path = os.path.join("models", self.name)
        self.model.save(file_path)
        
        # Save the Python metadata (excluding Spark model)
        spark_backup = self.spark
        model_backup = self.model
        self.spark = None
        self.model = None
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(self, f)
        self.spark = spark_backup
        self.model = model_backup
        print(f"Model and metadata exported to {file_path}")
        
    
    def predict(self, sids: np.ndarray, pids: np.ndarray):
        """
        Generates predictions for given (sid, pid) pairs using the trained ALS model.

        Input:
            sids (np.ndarray): Array of scientist IDs.
            pids (np.ndarray): Array of paper IDs.

        Outputs:
            np.ndarray: Array of predicted ratings in the original input order.
        """
        input_df = pd.DataFrame({
            'sid': sids,
            'pid': pids
        })
    
        # Add index to track original order
        input_df['index'] = range(len(input_df))
    
        # Convert to a Spark DataFrame
        test_data_spark = self.spark.createDataFrame(input_df.drop('index', axis=1))
    
        # Use the trained ALS model to make predictions
        predictions_spark = self.model.transform(test_data_spark)
    
        # Extract predictions as numpy array
        predictions_pd = predictions_spark.select("sid", "pid", "prediction").toPandas()
    
        # Merge with the input data to ensure predictions are aligned with the input order
        merged_df = input_df.merge(predictions_pd, on=['sid', 'pid'], how='left')
        merged_df = merged_df.sort_values('index')
    
        # Get predictions in the original order
        predictions_np = merged_df['prediction'].to_numpy()
        
        return np.clip(predictions_np, 1, 5)
    
    
    @classmethod
    def load(cls, name):
        """
        Loads a saved ALS model and metadata from disk.

        Input:
            name (str): Name of the model file (without extension).

        Outputs:
            AlternatingLeastSquares: Reconstructed model with Spark session and parameters.
        """
        path = os.path.join("models", name + ".pkl")
        model_path = os.path.join("models", name)

        with open(path, "rb") as f:
            obj = pickle.load(f)

        obj._init_spark()  # Recreate SparkSession
        obj.model = ALSModel.load(model_path)
        print(f"Loaded ALS model and metadata from {path}")
        return obj