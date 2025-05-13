# Collaborative Filtering - Computational Intelligence Lab FS 2025 - ML-Enjoyers
This repository contains our implementation for a Collaborative Filtering system for the Computational Intelligence Lab FS 2025 course. 
The objective is to predict scientist-paper ratings using collaborative filtering techniques.


## Task Overview

### Dataset
The dataset consists of two primary sources:

- **`train_ratings.csv`**: Contains explicit ratings provided by scientists for papers. Each row includes:
  - `sid`: Scientist ID
  - `pid`: Paper ID
  - `rating`: Integer score from 1 to 5
  > This dataset includes a total of **10,000 scientists** and **10,000 papers**

- **`train_tbr.csv`**  
  Contains **implicit feedback** in the form of a "to-be-read" list. These are papers that scientists expressed interest in but haven't rated. Each row has:
  - `sid`: Scientist ID (integer)
  - `pid`: Paper ID (integer)  
  > Each `(sid, pid)` pair represents a binary interaction - the presence of a row indicates the scientist is interested in that paper.

The ratings form a sparse user-item matrix, and the task is to predict the missing entries as accurately as possible.

### Evaluation
The primary evaluation metric is **Root Mean Squared Error (RMSE)**. We evaluate performance on a validation set created using a 75/25 random split of the training data


## Our Solution
We approached the matrix completion task using a mix of classical and modern recommender system models. Our models leverage both **explicit ratings** and **implicit feedback** derived from two sources:
- **Rated papers**: Scientists who rated papers implicitly reveal what they've interacted with.
- **Wishlisted papers**: Scientists who added papers to their "to-be-read" list express implicit interest.

After training the individual models, we combined their predictions using a **stacked ensemble**. Specifically, we used **Ridge regression** as a meta-learner to learn optimal weights to each base model’s output. The ensemble was trained on the validation set predictions and effectively balances the strengths of:
- ALS’s matrix factorization with global structure
- SVD++ and BFM’s rich handling of implicit data
- NCF's ability to model nonlinear interactions

### Models
| Model              | File         | Description |
|-------------------|--------------|-------------|
| **SVD++**         | `svdpp.py`   | SVD++ model using implicit signals from already rated and wishlisted papers with optional contrastive learning |
| **ALS**           | `als.py`     | Alternating Least Squares model using PySpark |
| **BFM**           | `bfm.py`     | Bayesian Factorization Machines with optional implicit features and ordinal regression |
| **NeuMF**         | `neuMF.py`   | Neural Collaborative Filtering combining GMF and MLP architectures |
| **Baselines**     | `baselines/` | Contains simple baseline methods such as SVD, SVP, SVT, and embedding dot-product to compare our results to |
| **Ensemble**      | `ensemble.py`| Ensemble of the models with Ridge regression for improved accuracy |



## Project Structure
```
.
├── als.py                  # ALS model
├── bfm.py                  # Bayesian Factorization Machine
├── svdpp.py                # SVD++ model
├── neuMF.py                # NCF model
├── ensemble.py             # Ridge regression ensemble of models
├── model.py                # Abstract base class for all models
├── training.py             # Orchestrates model training and evaluation for the individual models
├── utils.py                # Data loading, evaluation, preprocessing
├── baselines/              # Simple baseline models: SVD, SVP, SVT, DotProduct
├── models/                 # Folder for saved model parameters
├── data/                   # Contains training & sample submission data
├── submissions/            # Folder to store Kaggle-style CSV submissions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Running the Project
### Setup
Install required packages (Python 3.8+):

```bash
pip install -r requirements.txt
```
To run ALS (which relies on PySpark), ensure that Java is installed and accessible on your system. Additionally make sure **OpenJDK** is installed, and the `JAVA_HOME` environment variable is correctly set. For detailed installation instructions, refer to the [official PySpark setup guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).


### Train Individual Models
The `training.py` script orchestrates the training and validation of all individual models, including ALS, SVD++, NCF, and the various BFM variants. Each model is trained for 300 epochs and then stored into the `models/` directory for later use in ensembling. Validation is performed using a 75/25 split, and early stopping is applied for the PyTorch-based models (SVD++ and NCF), halting training when no further improvement was observed on the validation set after a specified patience period.

To train and export all models, run:

```bash
python training.py
```

If the `models/` directory already contains trained models, you can skip the training step and proceed directly to generating the final submission by running `ensemble.py`.


### Create Ensemble and Generate Final Predictions
The `ensemble.py` script constructs a stacked ensemble that integrates predictions from **ALS**, **SVD++** (with and without contrastive learning) **BFM** (in multiple configurations), and **NCF**. To leverage the complementary strengths of these models, we use a **Ridge regression meta-model** that learns optimal weights for combining their outputs, to improve overall prediction accuracy.

Once all base models are saved in the `models/` directory, you can generate the final submission by running:

```bash
python ensemble.py
```


