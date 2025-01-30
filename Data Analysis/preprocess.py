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

    
# plot_histogram(df, 'Glucose')
# plot_box(df,'Glucose')
count_cat(df,'Outcome')