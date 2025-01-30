from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ML.commons as commons
import ML.constants as constants
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_features(df):
    scaler = MinMaxScaler()
    df = pd.DataFrame(df)  # Ensure df is a DataFrame
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  
    return df_scaled


def tune_model_with_cross_val_predict(model, param, features, labels, k_values, binary_flag):
    try:
        # Input validation
        if not isinstance(features, (list, np.ndarray, pd.DataFrame)):
            raise TypeError("Features must be a list or numpy array.")
        if not isinstance(labels, (list, np.ndarray, pd.Series)):
            raise TypeError("Labels must be a list or numpy array.")
        if not isinstance(k_values, list) or not all(isinstance(k, int) for k in k_values):
            raise ValueError("k_values must be a list of integers.")
        if not isinstance(binary_flag, bool):
            raise TypeError("binary_flag must be a boolean.")

        best_model = None
        model_results = []

        for k in k_values:
            print(f"Performing {k}-fold Cross-Validation for {model.__class__.__name__}\n")
            try:
                # Perform grid search for hyperparameter tuning
                grid_search = perform_grid_search(model, param, features, labels, k)
                print(f"Best Hyperparameters for {model.__class__.__name__}: {grid_search.best_params_}")
                best_model = grid_search.best_estimator_

                # Use cross_val_predict for predictions
                try:
                    y_pred = cross_val_predict(best_model, features, labels, cv=k)
                    y_pred = commons.round_fractional_predictions(y_pred)
                except Exception as e:
                    raise RuntimeError(f"Error during cross_val_predict: {e}")

                # Compute evaluation metrics
                try:
                    eval_metrics = compute_eval_scores(y_pred, labels, binary_flag)
                    accuracy, f1, conf_metrics, precision, recall = eval_metrics
                except Exception as e:
                    raise RuntimeError(f"Error during evaluation metrics computation: {e}")

                model_results.append({
                    'Model': model.__class__.__name__,
                    'Accuracy': accuracy,
                    'F1 Score': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'K-Fold': k
                })

            except Exception as e:
                raise RuntimeError(f"Error during {k}-fold cross-validation for {model.__class__.__name__}: {e}")

        return best_model, model_results

    except ValueError as ve:
        raise ValueError(f"Input validation error: {ve}")
    except TypeError as te:
        raise TypeError(f"Type error in inputs: {te}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during tuning: {e}")
    pass

def get_model_and_params(model_dict, model_param, model_name, binary_flag):
    """
    Retrieves a machine learning model and its corresponding parameters based on the provided model name and binary flag.

    This function validates input arguments and retrieves the appropriate model and its parameters from the given dictionaries.
    If the model name or parameters are not found, or if invalid input is provided, appropriate exceptions are raised.

    Args:
        model_dict (dict): A dictionary containing model names as keys and model objects as values.
                            Maps model names to their respective machine learning models.
        model_param (dict): A dictionary containing model names as keys and parameter configurations for those models as values.
        model_name (str): The name of the model to retrieve. It should match one of the keys in both `model_dict` and `model_param`.
        binary_flag (bool): A boolean flag used to determine if the model is for binary classification. For models like "xgbc",
                                 it selects the appropriate model for binary classification (True or False).

    Returns:
        tuple: A tuple containing two elements:
            - model (object): The model object corresponding to the provided `model_name` and `binary_flag`.
            - param (dict): The parameters for the model, retrieved from `model_param`.

    Raises:
        TypeError: If `model_dict` is not a dictionary, `model_param` is not a dictionary, `model_name` is not a string,
                    or `binary_flag` is not a boolean.
        ValueError: If `binary_flag` is `None` or invalid for a model (e.g., for "xgbc"), or if the model or its parameters are not found.
        KeyError: If the `model_name` is not found in either `model_dict` or `model_param`, or if parameters for the model are not found.
        RuntimeError: If an unexpected error occurs, such as when the model or parameters are `None` after successful retrieval.

    Example Usage:
        model_dict = {
            "logreg": LogisticRegression(),
            "xgbc": {
                True: XGBClassifier(),
                False: XGBClassifier()
                }
            }

        model_param = {
            "logreg": {"C": 1.0, "solver": "lbfgs"},
            "xgbc": {"learning_rate": 0.01, "max_depth": 3}
            }

        # Get model and parameters for binary classification
        model, param = get_model_and_params(model_dict, model_param, "xgbc", True)

        print(model)  # XGBClassifier (binary)
        print(param)  # {'learning_rate': 0.01, 'max_depth': 3}

    Example with Edge Cases:
        # Model dictionary with only one model, but parameter dictionary missing the parameters for the model
        model_dict = {"logreg": LogisticRegression()}
        model_param = {}

        # This will raise a KeyError because parameters for 'logreg' are missing
        try:
            model, param = get_model_and_params(model_dict, model_param, "logreg", True)
        except KeyError as e:
            print(e)  # Key error in model or parameter retrieval: Parameters for model 'logreg' not found.

    Logic:
        1. **Input validation**:
            - Checks that `model_dict` and `model_param` are dictionaries.
            - Verifies that `model_name` is a string and `binary_flag` is a boolean, and that `binary_flag` is not `None`.

        2. **Model and parameter retrieval**:
            - Ensures that the `model_name` exists in both `model_dict` and `model_param`.
            - Retrieves the corresponding model and its parameters.

        3. **Special case for "xgbc" model**:
            - For "xgbc", it checks that the `binary_flag` exists in `model_dict[model_name]`. If it doesn't, it raises a `ValueError`.

        4. **Error handling**:
            - Raises specific exceptions for various error scenarios such as missing model names, incorrect types, or invalid binary flag.
            - If any unexpected errors occur (e.g., the model or parameters are `None`), a `RuntimeError` is raised.

    Author: Ritu
    """
    try:
        # Input validation
        if not isinstance(model_dict, dict):
            raise TypeError("model_dict must be a dictionary.")
        if not isinstance(model_param, dict):
            raise TypeError("model_param must be a dictionary.")
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if binary_flag is None:
            raise ValueError("binary_flag can't be None.")
        if not isinstance(binary_flag, bool):
            raise ValueError("binary_flag must be a boolean.")

        # Check if the model_name exists in either model_dict or model_param
        if model_name not in model_dict and model_name not in model_param:
            raise KeyError(f"Model {model_name} not found in model_dict or model_param.")

        # Retrieve the model
        if model_name not in model_dict:
            raise KeyError(f"Model {model_name} not found in model dictionary.")

        # Retrieve model parameters
        if model_name not in model_param:
            raise KeyError(f"Parameters for model {model_name} not found.")

        param = model_param[model_name]

        if model_name == "xgbc":
            if binary_flag not in model_dict[model_name]:
                raise ValueError(f"Binary flag '{binary_flag}' not valid for model '{model_name}'.")
            model = model_dict[model_name][binary_flag]
            # Check if we're using 'gblinear' and handle parameters accordingly
            if model.get_params().get('booster') == 'gblinear':
                # Remove or modify parameters that are not valid for 'gblinear'
                model.set_params(max_depth=None, min_child_weight=None)
        else:
            model = model_dict[model_name]

        if (model is None) or (param is None):
            raise RuntimeError(f"An unexpected error occurred")

        return model, param

    except KeyError as ke:
        raise KeyError(f"Key error in model or parameter retrieval: {ke}")
    except TypeError as te:
        raise TypeError(f"Type error in input arguments: {te}")
    except ValueError as ve:
        raise ValueError(str(ve))  # Use the exact error message
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    pass


def perform_hyperparameter_tuning(features, labels, model_dict=constants.MODEL_DICT, model_param=constants.MODEL_PARAMS, binary_flag=False):
    """
    Perform hyperparameter tuning for multiple machine learning models.

    This function reads feature matrices, labels, and additional data from files, preprocesses the data,
    and performs hyperparameter tuning for various models. It evaluates the models using cross-validation
    and saves the best-performing models and their results.

    Args:
        features_file (str): Path to the file containing the feature matrix.
        labels_file (str): Path to the file containing the labels.
        conc_file (str): Path to the file containing additional context or data.
        binary_flag (bool): Indicates if the problem is binary classification (True) or multi-class (False).

    Raises:
        ValueError: If features, labels, or additional data are missing or invalid.
        TypeError: If features, labels, or additional data are not lists or numpy arrays.
        RuntimeError: If any errors occur during file reading, data preprocessing, model tuning, or result saving.

    Workflow:
        1. **Data Loading**:
            - Reads the feature matrix, labels, and context data using `commons.read_features_set`.
            - Validates the presence and type of input data.

        2. **Data Preprocessing**:
            - Scales and preprocesses the feature matrix and labels using `preprocess_data`.

        3. **Model Hyperparameter Tuning**:
            - Iterates over models defined in `constants.MODEL_DICT` and their corresponding parameters in `constants.MODEL_PARAMS`.
            - Uses cross-validation and `tune_model_with_cross_val_predict` to find the best hyperparameters for each model.
            - Saves the best models using `save_model`.

        4. **Results Saving**:
            - Compiles results of hyperparameter tuning.
            - Saves these results as a CSV file in `constants.HYPER_TUNED_RESULTS_DIR`.

    Returns:
           None: This function does not return anything. It saves the results and models to disk.

    Example:
        >>> perform_hyperparameter_tuning("features.csv","labels.csv",binary_flag=True)

    Notes:
        - Ensure the input files exist and are correctly formatted.
        - The `binary_flag` determines the type of classification problem and adjusts the model parameters accordingly.
        - Check `constants.MODEL_DICT` and `constants.MODEL_PARAMS` for the supported models and their parameter grids.

    Dependencies:
        - `commons.read_features_set`: Reads feature sets from files.
        - `preprocess_data`: Preprocesses the features and labels.
        - `get_model_and_params`: Retrieves model instances and their parameter grids.
        - `tune_model_with_cross_val_predict`: Tunes models using cross-validation.
        - `save_model`: Saves trained models.
        - `commons.save_list_as_csv`: Saves lists as CSV files.

    Author: Ritu
    """
    if features is None or labels is None:
        raise ValueError("Features or labels not present")

    try:
        # Preprocess data
        scaled_features, labels = preprocess_data(features, labels)
        features = np.array(features)
        labels = np.array(labels)
    except Exception as e:
        raise RuntimeError(f"Error tuning model: {e}")

    try:
        # Hyperparameter tuning
        k_values = [5] # 7, 10
        best_models = {}
        results = []

        for model_name in model_dict.keys():
            print(f"Tuning hyperparameters for model: {model_name}")
            try:
                model, param = get_model_and_params(model_dict, model_param, model_name, binary_flag)
                features = np.array(features)
                labels = np.array(labels)
                best_model, model_results = tune_model_with_cross_val_predict(model, param, scaled_features, labels, k_values, binary_flag)
                best_models[model_name] = best_model
                model_params= best_model.get_params()
                results.extend(model_results)
                
                model_file_path = commons.create_file_path(constants.BEST_MODELS_DIR, model_name + "_hypertune.json")
                print(f"Saving {model_name.upper()} model")
                
                commons.save_params_as_json(model_file_path, model_params)
                
            except Exception as e:
                raise RuntimeError(f"Error tuning model {model_name}: {e}")

        try:
            commons.save_list_as_csv(constants.RESULTS_DIR, "hypertuningResults.csv", results)
        except Exception as e:
            raise RuntimeError(f"Error saving hyperparameter tuning results: {e}")

    except Exception as e:
        raise RuntimeError(f"An error occurred during hyperparameter tuning: {e}")
    return


# eval_metrics = compute_eval_scores(y_pred, labels, binary_flag)
# accuracy, f1, conf_metrics, precision, recall = eval_metrics
def compute_eval_scores(y_val, y_true, binary_flag, print_flag=True):
    """
    Computes and optionally prints evaluation metrics for classification models.

    This function calculates the following metrics:
    - Accuracy
    - F1-score
    - Confusion Matrix
    - Precision
    - Recall

    The metrics are computed based on whether the task is binary or multiclass,
    as determined by the `binary_flag`. Optionally, the results can be printed
    using a helper function.

    Parameters:
    - y_val (array-like):
        The predicted labels from the model.
    - y_true (array-like):
        The ground-truth labels.
    - binary_flag (bool):
        Indicator for the type of classification task.
        - `True` indicates binary classification.
        - Any other value indicates multiclass classification.
    - print_flag (bool, optional):
        If `True`, the computed evaluation scores will be printed using `print_eval_scores`.
        Default is `True`.

    Returns:
    - list:
        A list containing the following metrics in order:
        - `accuracy` (float): Accuracy of the predictions.
        - `f1` (float): F1-score based on the `binary_flag`.
        - `conf_matrix` (array-like): Confusion matrix.
        - `precision` (float): Precision score.
        - `recall` (float): Recall score.

    Raises:
        ValueError: If any input parameter is missing or invalid.

    Example Usage:
    >>> # Import necessary libraries
    >>> from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    >>> import numpy as np

    >>> # Example data
    >>> y_true = [0, 1, 0, 1, 1, 0, 1, 0]  # Ground truth labels
    >>> y_val = [0, 1, 0, 0, 1, 0, 1, 1]  # Predicted labels

    >>> # Binary classification example
    >>> metrics = compute_eval_scores(y_val, y_true, binary_flag=True)
    Model Evaluation Metrics:
    Accuracy: 0.75
    F1 Score: 0.80
    Precision: 0.75
    Recall: 0.86
    Confusion Matrix:
    3	1
    1	3

    >>> # Access individual metrics
    >>> accuracy, f1, conf_matrix, precision, recall = metrics
    >>> print(f"Accuracy: {accuracy}")
    Accuracy: 0.75

    Author: Ritu
    """
    try:
        # Validate input parameters
        check_missing_parameters(y_val=y_val, y_true=y_true, binary_flag=binary_flag)

        # Compute evaluation arguments
        avg_param, labels_arg = get_eval_args(binary_flag)

        # Compute evaluation scores
        accuracy = accuracy_score(y_val, y_true)
        f1 = f1_score(y_val, y_true, average=avg_param, zero_division=1)
        conf_matrix = confusion_matrix(y_val, y_true, labels=labels_arg)
        # Flatten confusion matrix to a single string
        flattened_conf_matrix = "; ".join(" ".join(map(str, row)) for row in conf_matrix)
        precision = precision_score(y_val, y_true, average=avg_param, zero_division=1)
        recall = recall_score(y_val, y_true, average=avg_param, zero_division=1)
        if print_flag:
            print_eval_scores(accuracy, f1, flattened_conf_matrix, precision, recall)
        return [accuracy, f1, flattened_conf_matrix, precision, recall]
    except ValueError as e:
        raise ValueError(f"ValueError during evaluation computation: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during evaluation computation: {e}")
    
def perform_grid_search(model, param, features, labels, k):
    """
    Perform grid search to find the best hyperparameters for a model.

    Args:
        model: The machine learning model to tune.
        param (dict): Hyperparameters for grid search.
        features (np.ndarray or list): Feature matrix.
        labels (np.ndarray or list): Target labels.
        k (int): Number of folds for cross-validation.

    Returns:
        GridSearchCV: Fitted GridSearchCV object with the best parameters.

    Raises:
        ValueError: If any of the inputs are invalid.
        RuntimeError: If grid search fails.

    Author: Ritu
    """
    try:
        # Input validation
        features = np.array(features)
        labels = np.array(labels)
        if not isinstance(features, (list, np.ndarray)):
            raise TypeError("Features must be a list or numpy array.")
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError("Labels must be a list or numpy array.")
        if not isinstance(k, int) or k <= 1:
            raise ValueError("k must be an integer greater than 1.")

        if not isinstance(param, dict):
            raise TypeError("param must be a dictionary of hyperparameters.")

        grid_search = GridSearchCV(model, param, cv=k, n_jobs=-1)

        try:
            grid_search.fit(features, labels)
        except Exception as e:
            raise RuntimeError(f"Error during grid search fitting: {e}")

        return grid_search

    except ValueError as ve:
        raise ValueError(f"Input validation error: {ve}")
    except TypeError as te:
        raise TypeError(f"Type error in inputs: {te}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during grid search: {e}")
    pass


def print_eval_scores(accuracy, f1, flattened_conf_matrix, precision, recall):
    """
    Prints evaluation scores and confusion matrix.
    """
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("Confusion Matrix:")
    print(flattened_conf_matrix)

    
def check_missing_parameters(**kwargs):
    """
    Checks if any of the provided parameters are missing or None.

    Args:
        **kwargs: Arbitrary keyword arguments representing the parameters to check.

    Raises:
        ValueError: If any parameter is missing or None. The error message will
                    include the names of the missing parameters.

    Author: Ritu
    """
    missing_params = [name for name, value in kwargs.items() if value is None]

    if missing_params:
        raise ValueError(f"The following parameters are missing or None: {', '.join(missing_params)}")
    return  missing_params


def get_eval_args(binary_flag):
    """
    Determines evaluation arguments based on whether the task is binary or multiclass classification.

    This function returns the appropriate averaging parameter (`avg_param`) and label arguments
    (`labels_arg`) for evaluation metrics such as precision, recall, or F1-score, depending on
    whether the classification task is binary or multiclass.

    Args:
        binary_flag (bool):
            Indicator for the type of classification task.
            - `True` indicates binary classification.
            - `False` or other values indicate multiclass classification.

    Returns:
        tuple:
            - avg_param (str): Averaging parameter for evaluation metrics.
                - `'binary'` for binary classification.
                - `'weighted'` for multiclass classification.
            - labels_arg (list or None):
                List of labels to include in the evaluation.
                - `[0, 1]` for binary classification.
                - `None` for multiclass classification (all labels included by default).

    Raises:
    ValueError: If the `binary_flag` is not a boolean value.
    Exception: If any unforeseen errors occur during the execution.

    Example Usage:
        >>> get_eval_args(True)
    ('binary', [0, 1])

    >>> get_eval_args(False)
    ('weighted', None)

    >>> get_eval_args("some string")
    Error: binary_flag must be a boolean value.

    Author: Ritu
    """
    try:
        print(f"binary flag: {binary_flag}")
        # Ensure binary_flag is a boolean value
        if binary_flag is None:
            raise ValueError("binary_flag can't be None for this function.")
        if not isinstance(binary_flag, bool):
            raise ValueError("binary_flag must be a boolean value.")
        if binary_flag:
            avg_param = 'binary'  # 'binary' for binary classification
            labels_arg = [0, 1]
        else:
            avg_param = 'weighted'  # 'weighted' for multiclass;
            labels_arg = None
        return avg_param, labels_arg

    except ValueError as e:
        # Catching the specific error for binary_flag type
        print(f"Error: {e}")
        raise  # Re-raise the exception to propagate it if necessary

    except Exception as e:
        # Catch any other unforeseen errors
        print(f"Unexpected error: {e}")
        raise  # Re-raise the exception to propagate it if necessary
    pass

def preprocess_data(x_train, y_train):
    """
    Balances and scales the training data and scales the test data for preprocessing.

    Args:
        - x_train (array-like or list): The training feature set.
        - y_train (array-like or list): The training labels.

    Returns:
        tuple: A tuple containing three elements:
            - x_train_scaled (array-like or list): The scaled and balanced training feature set.
            - y_train_bal (array-like or list): The balanced training labels.

    Raises:
        ValueError: If inputs are None, empty, or not in the expected format.
        RuntimeError: If there is an error during preprocessing.

    Author: Ritu
    """
    # Input validation
    if x_train is None or y_train is None:
        raise ValueError("Inputs cannot be None.")

    # Input type check logic
    # if not isinstance(x_train, (list, np.ndarray, pd.DataFrame)):
    #     raise ValueError("Input must be a list of lists, a NumPy array, or a pandas DataFrame.")
    #
    # if not (isinstance(y_train, (list, np.ndarray, pd.Series))):
    #     raise ValueError("Input must be a list of lists, a NumPy array, or a pandas DataFrame.")

    if len(x_train) == 0   or len(y_train) == 0:
        raise ValueError("Inputs cannot be empty.")

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have the same number of samples.")

    try:
        # Balance the training data
        x_train_bal, y_train_bal = get_balanced_feature_set(x_train, y_train)
        # Scale the features
        x_train_bal = pd.DataFrame(x_train_bal)
        x_train_scaled = scale_features(x_train_bal)
        return x_train_scaled, y_train_bal
    except Exception as e:
        raise RuntimeError(f"Error during data preprocessing: {e}")
    pass


from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

def get_balanced_feature_set(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the training dataset.
    
    Parameters:
        X_train (DataFrame): Feature training data.
        y_train (Series): Target training data.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train_resampled, y_train_resampled (tuple): Resampled training datasets.
    """
    # Print class distribution before SMOTE
    print("Before SMOTE:", Counter(y_train))

    # Apply SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print("After SMOTE:", Counter(y_train_resampled))
    
    return X_train_resampled, y_train_resampled

