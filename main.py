import Data_Analysis.preprocess
import ML.common
import pandas as pd
import ML.trainer
import ML , Data_Analysis, app
df = ML.common.read_csv('data','diabetes.csv')
# Data_Analysis.preprocess.stats(df,'data/stats.csv')
# Data_Analysis.preprocess.handle_missing(df, 'data/filteredCSV.csv')
filtered_df = pd.read_csv("data/filteredCSV.csv")
# Data_Analysis.preprocess.check_null(filtered_df)
#Data_Analysis.preprocess.plot_histogram(df,'Glucose')
#Data_Analysis.preprocess.classfier(df,'Glucose')

# Data_Analysis.preprocess.visualize_smote('data/diabetes.csv', 'data/train_smote.csv')
# scaledDf = ML.trainer.scale_features(filtered_df)
# scaledDf.to_csv("data/scaled.csv")
scaledDf = pd.read_csv("data/scaled.csv")
Data_Analysis.preprocess.smote('data/scaled.csv','Outcome')