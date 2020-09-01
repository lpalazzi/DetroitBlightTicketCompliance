import pandas as pd
import numpy as np
import datetime as dt

def blight_model():
    
    # Load data from the csv files

    train = pd.read_csv('train.csv', encoding='ISO-8859-1', dtype={'zip_code': np.str,'non_us_str_code': np.str,'grafitti_status': np.str})
    test = pd.read_csv('test.csv', encoding='ISO-8859-1')
    addresses = pd.read_csv('addresses.csv', encoding='ISO-8859-1')
    latlons = pd.read_csv('latlons.csv', encoding='ISO-8859-1')

    # Data preprocessing
    # --filter out Null values in compliance column
    # --separate into X_train, y_train, and X_test
    # --remove unnecessary features to prevent data leakage

    # Drop columns not present in test set
    train = train.drop(['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail'], axis=1)

    # Replace address columns with latitude and longitude 
    addresses = pd.merge(addresses, latlons, on='address', how='left')
    train = pd.merge(train, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'], axis=1)
    test = pd.merge(test, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'], axis=1)
    
    # Replace other spellings of Detroit with 'DETROIT'
    detroit_spellings = {'Deroit', 'DEROT', 'DERROIT', 'DERTOIT', 'DERTROIT', 'DEYROIT', 'DFETROIT', 'DKETROIT', 'DRTROIT', 'DTEROIT', 'DTROIT', 'DWETROIT', 'Det', 'DET ,', 'DET ROIT', 'DET,', 'DET,.', 'Det.', 'DET. MI.', 'DET.,', 'DET., MI.', 'det48234', 'DETAROIT', 'DETEOIT', 'DETEROIT', 'DETOIT', 'DETORIIT', 'DETORIT', 'DETOUR', 'DETOUR VILLAGE', 'DETR4OIT', 'DETREOIT', 'DETRIOIT', 'detriot', 'DETRIT', 'DETRIUT', 'DETRJOIT', 'Detro;it', 'Detrofit', 'Detroi', 'DETROIIT', 'DETROIOT', 'DETROIR', 'DETROIRT', 'DETROIS', 'Detroit', 'DETROIT  4', 'DETROIT,', 'DETROIT, MI.', 'DETROIT, MI. 48206', 'detroit,mi', 'DETROIT`', 'DETROIT1', 'DETROITdetroit', 'Detroitf', 'DETROITI', 'DETROITL', 'Detroitli', 'DETROITM', 'DETROITQ', 'detroitt', 'DETROITY', 'DETROIYT', 'DETROKT', 'DETROOIT', 'DETROPIT', 'detrorit', 'DETROT', 'DETROTI', 'DETROUIT', 'DETRPOT', 'DETRROIT', 'DETRTOI', 'detrtoit', 'DETTROIT', 'DETTROITM', 'DETYROIT'}
    train['city'] = train['city'].replace(to_replace=detroit_spellings, value='DETROIT')
    test['city'] = test['city'].replace(to_replace=detroit_spellings, value='DETROIT')

    # Convert dates to DateTime
    train['ticket_issued_date'] = pd.to_datetime(train['ticket_issued_date'], errors='coerce', yearfirst=True)
    train['hearing_date'] = pd.to_datetime(train['hearing_date'], errors='coerce', yearfirst=True)
    test['ticket_issued_date'] = pd.to_datetime(test['ticket_issued_date'], errors='coerce', yearfirst=True)
    test['hearing_date'] = pd.to_datetime(test['hearing_date'], errors='coerce', yearfirst=True)

    # New feature: days between ticket issue and hearing date
    train['date_delta'] = (train['hearing_date'] - train['ticket_issued_date']) / np.timedelta64(1, 'D')
    test['date_delta'] = (test['hearing_date'] - test['ticket_issued_date']) / np.timedelta64(1, 'D')

    # New feature: hearing_date day of the week
    
    # New feature: hearing_date hour of the day

    # New feature: mailing address not in USA (information in a few different features)
    train['non_usa'] = ( train['state'].isna() ) | ( train['non_us_str_code'].notna() ) | ( train['country'].apply(lambda x: False if ('USA' in str(x)) else True) )
    test['non_usa'] = ( test['state'].isna() ) | ( test['non_us_str_code'].notna() ) | ( test['country'].apply(lambda x: False if ('USA' in str(x)) else True) )

    # New feature: mailing address not in Michigan
    train['non_michigan'] = train['state'].apply(lambda x: False if ('MI' in str(x)) else True)
    test['non_michigan'] = test['state'].apply(lambda x: False if ('MI' in str(x)) else True)
    
    # Drop some unnecessary columns in both train and test sets
    train = train.drop(['mailing_address_str_number', 'mailing_address_str_name', 'zip_code', 'non_us_str_code', 'country', 'state', 'ticket_issued_date', 'hearing_date'], axis=1) 
    
    # Free up dataframes no longer used
    del latlons
    del addresses

    # Select classifier models to test
    # --make a custom class that will keep track of all the models and their hyperparameters, and store the evaluation results throughout the process

    # Use cross validation with all of the models

    # Re-train the best model using full training set and evaluate

    return # Your answer here

blight_model()