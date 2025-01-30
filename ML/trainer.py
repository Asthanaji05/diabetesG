from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def scale_features(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  # Apply transformation and keep column names
    return df_scaled
