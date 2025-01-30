import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import csv
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ML.common import read_csv



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

def smote(data_path, col):
    X_train, X_test, y_train, y_test = apply_smote(data_path, col)
    # Combine X and y into one DataFrame
    train_df = pd.DataFrame(X_train)
    train_df[col] = y_train  # Add target column

    test_df = pd.DataFrame(X_test)
    test_df[col] = y_test  # Add target column

    # Save the new datasets
    train_df.to_csv("data/train_smote.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
def visualize_smote(before,after):
    bf = pd.read_csv(before)
    af = pd.read_csv(after)
    bf_plt = coorelation(bf)
    af_plt = coorelation(af)

def coorelation(df):
    corr = df.corr()
    #plot corr
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.plot()
    plt.show()
    return plt
    # print(corr)
    
def stats(df,filepath):
    df.describe().to_csv(filepath)  
    print('sucess')  
    return    
def avgs(df, col):
    # avgs of col where outcome is 0
    df_0 = df[df['Outcome'] == 0]
    avg_0 = df_0[col].mean()
    # avgs of col where outcome is 1
    df_1 = df[df['Outcome'] == 1]
    avg_1 = df_1[col].mean()
    return avg_0,avg_1
    
def handle_missing(df, filepath):
    Columns= ["Glucose","BloodPressure","SkinThickness","Insulin","BMI" ]
    for col in Columns:
        avg0 , avg1 = avgs(df , col)
        # fill avgs in place of zeros as per [outcome]
        df.loc[(df['Outcome'] == 0) & (df[col] == 0), col] = avg0
        df.loc[(df['Outcome'] == 1) & (df[col] == 0), col] = avg1
    df.to_csv(filepath, index=False)  # File save karega given path pe
    print("Done !")
    return df

def check_null(df):
        # Check for missing values
    print(df.isnull().sum())
    # check for zeros in col except outcome
    for col in df.columns:
        if col != 'outcome':
            print(f"Column: {col} :  {df[col].eq(0).sum()}")
    
def classfier(df,col):   
    mean = df[col].mean() # mean
    # standard deviation
    std = df[col].std()
    # classifies as "low" , "medium" , "high" based on mean and std
    classification = df[col].apply(lambda x : 'High' if x> mean+std else 'low' if x<mean-std else 'Medium')
    print(classification.value_counts().to_dict())
    

    
    
    


