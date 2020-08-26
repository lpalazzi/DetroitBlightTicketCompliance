import pandas as pd
import numpy as np

def blight_model():
    
    # Load data from the csv files

    train = pd.read_csv("train.csv", encoding="ISO-8859-1", dtype={'zip_code': np.str,'non_us_str_code': np.str,'grafitti_status': np.str})
    test = pd.read_csv("test.csv", encoding="ISO-8859-1")
    addresses = pd.read_csv("addresses.csv", encoding="ISO-8859-1")
    latlons = pd.read_csv("latlons.csv", encoding="ISO-8859-1")

    # Data preprocessing
    # --filter out Null values in compliance column
    # --separate into X_train, y_train, and X_test
    # --remove unnecessary features to prevent data leakage

    # Drop columns not present in test set
    train = train.drop(['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail'], axis=1)
    
    # Replace address columns with latitude and longitude 
    addresses = pd.merge(addresses, latlons, on='address', how='left')
    train = pd.merge(train, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'])
    test = pd.merge(test, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'])
    
    # Free up dataframes no longer used
    del latlons
    del addresses


    

    # Select classifier models to test
    # --make a custom class that will keep track of all the models and their hyperparameters, and store the evaluation results throughout the process

    # Use cross validation with all of the models

    # Re-train the best model using full training set and evaluate

    return # Your answer here

blight_model()