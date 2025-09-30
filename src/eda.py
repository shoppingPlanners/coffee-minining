# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df: pd.DataFrame, figsize=(10,8)):
    """Plot correlation heatmap"""
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """Plot target variable distribution"""
    sns.countplot(x=target_col, data=df)
    plt.title(f"{target_col} Distribution")
    plt.show()

def summary_statistics(df: pd.DataFrame):
    """Print basic statistics"""
    print(df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
