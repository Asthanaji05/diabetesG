import pandas as pd
import ML.trainer
import ML.constants as constants
from Data_Analysis import preprocess  # Ensure correct import
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the specific ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Handle missing values if needed
# preprocess.handle_missing(df, 'data/filteredCSV.csv')

# Save filtered data
df.to_csv("data/filteredCSV.csv", index=False)

# Load filtered data
filtered_df = pd.read_csv("data/filteredCSV.csv")

# Apply SMOTE on scaled data
preprocess.smote("data/filteredCSV.csv", "Outcome")

# Load SMOTE-balanced data
smote_df = pd.read_csv("data/train_smote.csv")

# Separate features and labels
features = smote_df.drop(columns=['Outcome']).values
labels = smote_df['Outcome'].values

# Scale the features
scaled_features = ML.trainer.scale_features(pd.DataFrame(features))

# Save scaled data
scaled_features.to_csv("data/scaled.csv", index=False)

# Convert to NumPy arrays
features = scaled_features.to_numpy()
labels = labels.astype(int)  # Ensure labels are integer type

# Train models
ML.trainer.perform_hyperparameter_tuning(
    features, labels,
    model_dict=constants.MODEL_DICT,
    model_param=constants.MODEL_PARAMS,
    binary_flag=True
)