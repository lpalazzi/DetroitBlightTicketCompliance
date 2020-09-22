import pandas as pd
import numpy as np
import datetime as dt

def process_data(df, addresses):
    # ------------------------------
    # Data preprocessing
    # ------------------------------

    # Replace address columns with latitude and longitude 
    df = pd.merge(df, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'], axis=1)
    
    # Replace other spellings of Detroit with 'DETROIT'
    detroit_spellings = {'Deroit', 'DEROT', 'DERROIT', 'DERTOIT', 'DERTROIT', 'DEYROIT', 'DFETROIT', 'DKETROIT', 'DRTROIT', 'DTEROIT', 'DTROIT', 'DWETROIT', 'Det', 'DET ,', 'DET ROIT', 'DET,', 'DET,.', 'Det.', 'DET. MI.', 'DET.,', 'DET., MI.', 'det48234', 'DETAROIT', 'DETEOIT', 'DETEROIT', 'DETOIT', 'DETORIIT', 'DETORIT', 'DETOUR', 'DETOUR VILLAGE', 'DETR4OIT', 'DETREOIT', 'DETRIOIT', 'detriot', 'DETRIT', 'DETRIUT', 'DETRJOIT', 'Detro;it', 'Detrofit', 'Detroi', 'DETROIIT', 'DETROIOT', 'DETROIR', 'DETROIRT', 'DETROIS', 'Detroit', 'DETROIT  4', 'DETROIT,', 'DETROIT, MI.', 'DETROIT, MI. 48206', 'detroit,mi', 'DETROIT`', 'DETROIT1', 'DETROITdetroit', 'Detroitf', 'DETROITI', 'DETROITL', 'Detroitli', 'DETROITM', 'DETROITQ', 'detroitt', 'DETROITY', 'DETROIYT', 'DETROKT', 'DETROOIT', 'DETROPIT', 'detrorit', 'DETROT', 'DETROTI', 'DETROUIT', 'DETRPOT', 'DETRROIT', 'DETRTOI', 'detrtoit', 'DETTROIT', 'DETTROITM', 'DETYROIT'}
    df['city'] = df['city'].replace(to_replace=detroit_spellings, value='DETROIT')

    # Convert dates to DateTime
    df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date'], errors='coerce', yearfirst=True)
    df['hearing_date'] = pd.to_datetime(df['hearing_date'], errors='coerce', yearfirst=True)

    # New feature: days between ticket issue and hearing date
    df['date_delta'] = (df['hearing_date'].fillna(dt.datetime(year=2200,month=1,day=1)) - df['ticket_issued_date']) / np.timedelta64(1, 'D')

    # New feature: mailing address not in USA (information in a few different features)
    df['non_usa'] = ( df['state'].isna() ) | ( df['non_us_str_code'].notna() ) | ( df['country'].apply(lambda x: False if ('USA' in str(x)) else True) )

    # New feature: mailing address not in Michigan
    df['non_michigan'] = df['state'].apply(lambda x: False if ('MI' in str(x)) else True)
    
    # Drop some unnecessary columns
    df = df.drop(['mailing_address_str_number', 'mailing_address_str_name', 'zip_code', 'non_us_str_code', 'country', 'state', 'ticket_issued_date', 'hearing_date'], axis=1)
    
    return df


def blight_model():
    
    # Load data from the csv files
    train = pd.read_csv('train.csv', encoding='ISO-8859-1', dtype={'zip_code': np.str,'non_us_str_code': np.str,'grafitti_status': np.str})
    test = pd.read_csv('test.csv', encoding='ISO-8859-1')
    addresses = pd.merge(pd.read_csv('addresses.csv', encoding='ISO-8859-1'), pd.read_csv('latlons.csv', encoding='ISO-8859-1'), on='address', how='left')

    # Drop the training set columns not present in test set
    train = train.drop(['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail'], axis=1)

    # Pre-preocess the train and test sets
    train = process_data(train, addresses)
    test  = process_data(test,  addresses)

    # Free up dataframes no longer used
    del addresses

    # Select classifier models to test
    # --make a custom class that will keep track of all the models and their hyperparameters, and store the evaluation results throughout the process

    # Use cross validation with all of the models

    # Re-train the best model using full training set and evaluate

    print (train['date_delta'].head(15))

    return # Your answer here

blight_model()