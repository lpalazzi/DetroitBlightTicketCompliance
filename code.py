import pandas as pd
import numpy as np

def blight_model():
    
    # Load data from the csv files

    train = pd.read_csv("train.csv", encoding="ISO-8859-1", )
    #test = pd.read_csv("test.csv", encoding="ISO-8859-1")

    cols = train.columns
    types = train.dtypes
    print (types)

    # Data preprocessing
    # --filter out Null values in compliance column
    # --separate into X_train, y_train, and X_test
    # --remove unnecessary features to prevent data leakage
    
    # Select classifier models to test
    # --make a custom class that will keep track of all the models and their hyperparameters, and store the evaluation results throughout the process

    # Use cross validation with all of the models

    # Re-train the best model using full training set and evaluate
    
    return # Your answer here

blight_model()