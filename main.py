import Data_Analysis.preprocess
import ML.common
import ML , Data_Analysis, app
df = ML.common.read_csv('data','diabetes.csv')
Data_Analysis.preprocess.plot_histogram(df,'Glucose')