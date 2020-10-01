import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def process_data(df, addresses):
    # ------------------------------
    # Data preprocessing
    # ------------------------------

    # Replace address columns with latitude and longitude 
    df = pd.merge(df, addresses, on='ticket_id', how='left').drop(['violation_street_number', 'violation_street_name', 'violation_zip_code'], axis=1)

    # Remove where lat or lon is null
    df['lat'] = df['lat'].fillna(0)
    df['lon'] = df['lon'].fillna(0)

    # New feature: mailing address not in USA (information in a few different features)
    df['non_usa'] = ( df['state'].isna() ) | ( df['non_us_str_code'].notna() ) | ( df['country'].apply(lambda x: False if ('USA' in str(x)) else True) )

    # New feature: mailing address not in Michigan
    df['non_michigan'] = df['state'].apply(lambda x: False if ('MI' in str(x)) else True)

    # Pick the columns to include in the final DataFrame
    df_final = df[['ticket_id', 'judgment_amount', 'lat', 'lon', 'non_usa', 'non_michigan', 'compliance']]

    return df_final


def blight_model():
    
    # Load data from the csv files
    train = pd.read_csv('train.csv', encoding='ISO-8859-1', dtype={'zip_code': np.str,'non_us_str_code': np.str,'grafitti_status': np.str})
    test = pd.read_csv('test.csv', encoding='ISO-8859-1')
    addresses = pd.merge(pd.read_csv('addresses.csv', encoding='ISO-8859-1'), pd.read_csv('latlons.csv', encoding='ISO-8859-1'), on='address', how='left')

    # Drop the training set columns not present in test set
    train = train.drop(['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail'], axis=1)

    # Drop the training set rows where "not responsible" (tickets where the violators were found not responsible are not considered during evaluation)
    train = train.dropna(subset=['compliance'])

    # Add an empty column to test set for 'compliance'
    test['compliance'] = np.nan

    # Convert compliance floats to integer
    train['compliance'] = train['compliance'].astype(int)

    # Pre-preocess the train and test sets
    train = process_data(train, addresses)
    test  = process_data(test,  addresses)

    # Split into X, y
    X_train = train[['judgment_amount', 'lat', 'lon', 'non_usa', 'non_michigan']]
    X_test  = test [['ticket_id', 'judgment_amount', 'lat', 'lon', 'non_usa', 'non_michigan']]
    y_train = train['compliance']
    y_test  = pd.Series(data=np.nan, index=test['ticket_id'])

    # Free up dataframes no longer used
    del addresses

    # Create and train model
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)

    for ticket_id, probability in y_test.iteritems():
        test_sample = X_test[X_test['ticket_id'] == ticket_id].drop(['ticket_id'], axis=1)
        y_test.loc[ticket_id] = model.predict_proba(test_sample)[:,1]

    return y_test
