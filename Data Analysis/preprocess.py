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

# add a histogram of the 'Glucose' column
plt.hist(df['Glucose'], bins=10, color='blue', edgecolor='black')
