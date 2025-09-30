# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with mode for categorical columns"""
    categorical_cols = ['Gender', 'Country', 'Sleep_Quality', 'Occupation', 'Smoking', 'Alcohol_Consumption']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def encode_categorical(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    """Label encode categorical columns"""
    if categorical_cols is None:
        categorical_cols = ['Gender', 'Country', 'Sleep_Quality', 'Occupation', 'Smoking', 'Alcohol_Consumption']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Example: BMI category encoding"""
    if 'BMI' in df.columns:
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100], labels=[0,1,2,3])
    return df

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
