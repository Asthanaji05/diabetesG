import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ML.common import read_csv
# use data/diabetes.csv
df = read_csv("data", "diabetes.csv")
print(df.head())


def plot_histogram(df, col):
    # add a histogram of the 'Glucose' column
    plt.hist(df[col], bins=10, color='limegreen', edgecolor='black')
    plt.plot()
    plt.show()

def plot_box(df,col):
    # box plot of Glucose
    #green color
    sns.boxplot(x=df[col], color='limegreen')
    plt.plot()
    plt.show()
    
def count_cat(df,col):
    # count zeros and ones
    print(df[col].value_counts())
    #plot bar graph
    sns.countplot(x=col, data=df)
    plt.plot()
    plt.show()

    
# plot_histogram(df, 'Glucose')
# plot_box(df,'Glucose')
# count_cat(df,'Outcome')

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

def apply_smote(data_path, target_column, test_size=0.2, random_state=42):
    """
    Applies SMOTE to balance the dataset.
    
    Parameters:
        data_path (str): Path to the CSV file.
        target_column (str): Name of the target column.
        test_size (float): Test dataset split ratio.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train_resampled, X_test, y_train_resampled, y_test (tuple): Resampled training and test datasets.
    """

    # Load dataset
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Print class distribution before SMOTE
    print("Before SMOTE:", Counter(y_train))

    # Apply SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print("After SMOTE:", Counter(y_train_resampled))

    return X_train_resampled, X_test, y_train_resampled, y_test

# data_path = "data/diabetes.csv"  # Update with the correct path
# target_column = "Outcome"  # Update with the correct target column name
# X_train, X_test, y_train, y_test = apply_smote(data_path, target_column)

# function for correlation among variables

def coorelation(df):
    corr = df.corr()
    #plot corr
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.plot()
    plt.show()
    # print(corr)
    
def stat(df,col):
    # mean , mode , min , max 
    print(df[col].mean())
    
    


