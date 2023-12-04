#model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

filterwarnings("ignore")

def RandomForest(data):
    
    datacols = data.columns

    df = pd.read_csv('HR_DS.csv')
    df = df.drop(['Over18', 'EmployeeNumber','EmployeeCount','StandardHours','MonthlyIncome' ,'YearsInCurrentRole', 'YearsWithCurrManager'],axis=1)

    cat = df.select_dtypes('object')
    for col in cat.columns:
        n = 2
        for i in cat[col].unique():
            df[col] = df[col].replace(i,n)
            n += 1
        
    # normalizing
    scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
    norm = scaler.fit_transform(df)
    norm_df = pd.DataFrame(norm,columns=df.columns)
    
    X = pd.DataFrame(norm_df.drop(columns='Attrition'))
    Y = pd.DataFrame(norm_df.Attrition).values.reshape(-1, 1)
    
    # Drop columns that are not in the specified list
    X = X.drop(columns=X.columns.difference(datacols))

    # Train first and doing oversampling to reduce the imbalance problem
    x_train, x_test, y_train, y_test = train_test_split(X ,Y ,test_size = 0.2 , random_state = 0)
    smote = SMOTE(random_state=0)
    smote_train, smote_target = smote.fit_resample(x_train,y_train)

    # train for Random Forest
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    rfc = RandomForestClassifier()
    rfc = rfc.fit(smote_train , smote_target)
    
    data = data.reindex(columns=X.columns)
    
    y_pred = rfc.predict(data)

    return y_pred