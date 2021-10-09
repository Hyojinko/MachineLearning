import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings(action='ignore')

def Missingvalue(df):
    # check missing value
    # only 'totla_bedrooms' has missing values, fill median
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)


    X = df.drop(['median_house_value'], axis = 1)     # feature longitude, latitude, ..., ocean_proximity
    y = df.iloc[:, -2].copy()                         # target median_house_value

    num_cols = X.select_dtypes(include = ['int64', 'float64']).columns.to_list()    #numerical value
    cat_cols = X.select_dtypes(include = ['object']).columns.to_list()              # categorical value
    
    return X,y,num_cols,cat_cols



def Combination_List(X, y,scale_feature,encode_feature):

    # 1. Scaler List : Standard, MinMax, maxAbs, Robust
    standard = StandardScaler()
    minMax = MinMaxScaler()
    maxAbs = MaxAbsScaler()
    robust = RobustScaler()
    scalers = {"standard scaler": standard, "minMax scaler": minMax, "maxAbs scaler": maxAbs, "robust scaler": robust}

    # 2. Encoder List : Label, One-hot
    label = LabelEncoder()
    oneHot = OneHotEncoder()
    encoders={"label encoder": label,"one-hot encoder": oneHot}


    return X, y,scale_feature,encode_feature, scalers, encoders


def preprocessing(X,y,scale_feature,encode_feature, scalers, encoders):
    
    # combinations of scaler and encoder
    # scalers
    for scaler_key, scaler in scalers.items():
        
        X[scale_feature] = scaler.fit_transform(X[scale_feature])
        print("\n-----------------------------------------------------------------------")
        print(f'<   scaler: {scaler_key}    >')
        
        # encoders
        for encoder_key, encoder in encoders.items():
            # label encoder
            if encoder_key=="label encoder":
                X_1=X.copy()
                def label_encoder(data):
                    for i in encode_feature:
                        data[i] = encoder.fit_transform(data[i])
                    return data
                X_label=label_encoder(X_1)

                cleaned_df=pd.concat([X_label,y],axis=1)                           
                print(f'\n      <   encoder: {encoder_key}   >')

                print(cleaned_df.head())
                               
            # Onegit-hot encoder
            if encoder_key=="one-hot encoder":
                X_onehot=pd.get_dummies(X)
                               
                cleaned_df=pd.concat([X_onehot,y],axis=1)                                
                print(f'\n      <   encoder: {encoder_key}   >')

                print(cleaned_df.head())      

    return cleaned_df


df=pd.read_csv("../Dataset/housing.csv")


# preprocessing
X,y,scale_feature,encode_feature = Missingvalue(df)

X, y,scale_feature,encode_feature, scalers, encoders= Combination_List(X, y,scale_feature,encode_feature)

cleaned_df= preprocessing(X,y,scale_feature,encode_feature,scalers, encoders)
