from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import ML.commons as commons
import ML.constants as constants
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA , IncrementalPCA
from sklearn.preprocessing import LabelEncoder


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
                    accuracy, f1, conf_metrics, precision, recall , report= eval_metrics
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

        return best_model, model_results , report

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
        k_values = [3,5,7,10] # 7, 10
        best_models = {}
        results = []

        for model_name in model_dict.keys():
            print(f"Tuning hyperparameters for model: {model_name}")
            try:
                model, param = get_model_and_params(model_dict, model_param, model_name, binary_flag)
                features = np.array(features)
                labels = np.array(labels)
                best_model, model_results,report = tune_model_with_cross_val_predict(model, param, scaled_features, labels, k_values, binary_flag)
                best_models[model_name] = best_model
                model_params= best_model.get_params()
                results.extend(model_results)
                
                model_file_path = commons.create_file_path(constants.BEST_MODELS_DIR, model_name + "_hypertune.json")
                print(f"Saving {model_name.upper()} model")
                
                report_file_path = commons.create_file_path(constants.BEST_MODELS_DIR, model_name + "_report.json")
                print(f"Saving {model_name.upper()} report")
                commons.save_params_as_json(report_file_path, report)

                print(f"Saved classification report for {model_name.upper()} model")
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
        
        accuracy = accuracy_score(y_true, y_val)
        f1 = f1_score(y_true, y_val, average=avg_param, zero_division=1)
        conf_matrix = confusion_matrix(y_true, y_val, labels=labels_arg)
        # Flatten confusion matrix to a single string
        flattened_conf_matrix = "; ".join(" ".join(map(str, row)) for row in conf_matrix)
        precision = precision_score(y_true, y_val, average=avg_param, zero_division=1)
        recall = recall_score(y_true, y_val, average=avg_param, zero_division=1)
        # classification report 
        report = classification_report(y_true, y_val , output_dict=True)
        if print_flag:
            print_eval_scores(accuracy, f1, flattened_conf_matrix, precision, recall)
        return [accuracy, f1, flattened_conf_matrix, precision, recall , report]
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

def save_model_as_json(model, model_path, conc_flag=False, area_flag=False, hypertune_flag=False):
    """
    Saves a given machine learning model to a specified directory as a pickle file.

    Args:
        model (object): The machine learning model to be saved. This can be any object that is serializable with pickle.
        model_name (str): The file name (with or without extension) to save the model as. Must be a non-empty string.
        conc_flag (bool, optional): If True, the model will be saved in the constants.CONC_MODELS_DIR directory.
        area_flag (bool, optional): If True, the model will be saved in the constants.AREA_MODELS_DIR directory.
                                    Ignored if `conc_flag` is True.
        hypertune_flag (bool): If True, selects the hypertuning models directory (constants.HYPER_TUNED_MODELS_DIR).
        If all the flags are false, model is saved in the constants.MODELS_DIR
    Returns:
        bool: True if the model is saved successfully, False otherwise.

    Raises:
        ValueError: If `model_name` is not a valid non-empty string.
        TypeError: If the `model` object is not serializable by pickle.
        FileNotFoundError: If the directory or file path is invalid.
        IOError: If an I/O error occurs while writing the file.

    Notes:
        - The function uses constants (`CONC_MODELS_DIR`, `AREA_MODELS_DIR`, `MODELS_DIR`) to determine the save directory.
        - If neither `conc_flag` nor `area_flag` is set, the model is saved in the default models directory.

    Author: Ritu
    """
    try:
        #model_dir = get_model_dir(conc_flag, area_flag, hypertune_flag)
        # Validate the model_name
        #if not isinstance(model_name, str) or not model_name.strip():
        #    raise ValueError("Model name must be a non-empty string.")

        # print(f"model_path: {model_dir} {model_name}")

        # # Ensure the directory exists
        # os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        # model_path = commons.create_file_path(model_dir, model_name.lower())
        # if not model_path.endswith(".json"):
        #     model_path = model_path + ".json"
        #print(f"model_path: {model_dir} {model_name}")

        # Ensure the directory exists
        #os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        if not model_path.endswith(".json"):
            model_path = model_path.lower() + ".json"
        #model_path = commons.create_file_path(model_dir, model_name)

        # Save the model
        model.save_model(model_path)
        print(f"Model saved successfully at: {model_path}")
        return True

    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except TypeError as te:
        print(f"TypeError: The model is not serializable. {te}")
        raise
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: The directory or file path is invalid. {fnfe}")
        raise
    except IOError as ioe:
        print(f"IOError: An I/O error occurred while writing the file. {ioe}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def train_model(x_train, y_train, model_name, binary_flag, voc_name=None, conc_flag=False, area_flag=False):
    """
    Trains classifiers defined in a dictionary on the provided dataset and saves the trained models.

    This function loops over a dictionary of predefined models, trains each model on the provided
    dataset, and saves them with appropriate filenames.

    Args:
        x_train (array-like or DataFrame):
            The feature set of the training data. Can be a NumPy array, Pandas DataFrame,
            or any format compatible with the `fit()` method of the classifiers.
        y_train (array-like or Series):
            The target labels corresponding to `x_train`. Must be a 1-dimensional array or
            Pandas Series containing class labels.
        model_name (str):
            name of the model with which the training needs to be performed.
            model is fetched from constants.BEST_MODELS based on this model_name as key.
        binary_flag (bool):
            specifies if it is binary classification (if flag is True) or multi-class.
            Determines a particular xgbc model parameter configuration.
        voc_name (str, optional):
            The VOC (Volatile Organic Compound) name used to prefix the model name if `conc_flag` is True.
            Defaults to None.
        conc_flag (bool, optional):
            If True, indicates the model is concentration-specific, and the VOC name is included in the model name.
            Defaults to False.  
    Raises:
        ValueError: If `conc_flag` is True but `voc_name` is not provided or is empty.

    Example Usage:
        >>> from sklearn.datasets import make_classification
        >>> x_train, y_train = make_classification(n_samples=100, n_features=10, random_state=42)
        >>> trained_models = train_model(x_train, y_train, voc_name="benzene", conc_flag=True)

    Author: Ritu
    """
    # Check if BEST_MODELS_DICT exists and has elements
    # if not hasattr(constants, "BEST_MODELS_DICT") or not constants.BEST_MODELS_DICT:
    #    raise ValueError("BEST_MODELS_DICT is not defined or is empty in constants.")

    models_dict = constants.BEST_MODELS_DICT
    model = None
    try:
        # Validate voc_name when conc_flag is True
        if conc_flag and voc_name is None:
            raise ValueError("voc_name must be provided when conc_flag is True.")

        # Validate the model_name and binary_flag
        if model_name not in models_dict.keys():
            raise ValueError(f"Invalid model name: {model_name}. Model name should be among: {models_dict.keys()}")
        if not isinstance(binary_flag, bool):
            raise ValueError(f"binary_flag must be a boolean (True or False) for model {model_name}.")
        if model_name == 'xgbc':
            if binary_flag not in models_dict[model_name]:
                raise ValueError(f"Invalid binary_flag: {binary_flag} for model {model_name}.")
            model = models_dict[model_name][binary_flag]
        else:
            model = models_dict[model_name]

        # Validate and preprocess x_train
        if x_train is None or y_train is None:
            raise ValueError("Training data (x_train, y_train) cannot be None.")

        if not isinstance(x_train, (list, pd.DataFrame, np.ndarray)):
            raise ValueError("x_train must be a list of lists, a NumPy array, or a pandas DataFrame.")

        # Ensure y_train is a list
        if not isinstance(y_train, (list, pd.Series, np.ndarray)):
            raise ValueError("y_train must be a list, NumPy array, or pandas Series.")

        # Ensure that x_train and y_train are not empty
        if len(x_train) == 0 or len(y_train) == 0:
            raise ValueError("x_train and y_train cannot be empty.")

        # Ensure that x_train and y_train have matching lengths
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same number of samples.")

        model.fit(x_train, y_train)
        model_params = model.get_params()
        # commons.save_params_as_json(model, model_name, conc_flag)

        if not conc_flag:
            model_dir = constants.BEST_MODELS_DIR
        else:
            model_dir = constants.CONC_MODELS_DIR
        # Generate model name and save
        model_file_name = (commons.get_model_name_prefix(conc_flag, area_flag, voc_name) + model_name)
        model_file_path = commons.create_file_path(model_dir, model_file_name + ".json")
        print(f"Saving {model_name.upper()} model")
        if model_name == 'xgbc' or model_name == 'cat':
            # Attempt to save the model
            if not save_model_as_json(model, model_file_path, conc_flag, area_flag):
                raise Exception(f"Failed to save model {model_name}.")
        else:
            commons.save_params_as_json(model_file_path, model_params)

    except Exception as e:
        # Handle any errors during file creation or saving
        print(f"Error saving model parameters: {e}")
    return model

def incremental_train(x_new, y_new, model_name, binary_flag, voc_name=None, conc_flag=False, area_flag=False):
    model = None
    if model_name == 'xgbc':
        best_models_dir = constants.BEST_MODELS_DIR
        model_path = commons.create_file_path(best_models_dir, model_name+".json")
        loaded_model = commons.load_json_model(model_path, binary_flag)
        # Create DMatrix from data
        d_new = loaded_model.DMatrix(x_new, label=y_new)
        # Continue training the model using the new data
        params = constants.BEST_MODELS_PARAM[model_name][binary_flag]
        model = loaded_model.train(params, d_new, num_boost_round=100, xgb_model=loaded_model)
        model.save_model('retrained_xgbc_.json')
    else:
        print("incremental support is available for xbgc only in this version")
    return model


# savgol_filter for IOT senser related data
# you have to find optimum windo-length 

import shap

def calculate_shap(model, x_train, explainer="tree"):
    """
    This function calculates the shap values for given X(input data) and a model

    Parameters
    - model(ML/DL model): a trained model
    - x_train(dataframe): input data or the feature matrix on which a ml model can be trained

    Returns
    - shap_values of dimension(n_samples,n_features,n_classes)
    - feature_names of the input data/feature matrix
    """
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train, columns=[i for i in range(len(x_train[0]))])
    if explainer == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    shap_values = explainer(x_train)
    shap_values = shap_values.values
    feature_names = x_train.columns
    return shap_values, feature_names


def aggregate_top_features_per_class_shap(shap_values, feature_names, top_n=5):
    """
    This function identifies the top N features contributing to predictions for each class based on SHAP (SHapley Additive exPlanations) values.

    Parameters:
    - shap_values (ndarray): A 3D array of SHAP values with dimensions (n_samples, n_features, n_classes).
    - feature_names (list): feature names corresponding to the SHAP values.
    - top_n (int, optional): The number of top features to extract for each class. Default is 10.

    Returns:
    - top_10_features_dict (dict): A dictionary where keys are class indices and values are arrays of the top top_n feature names for each class.
    - top_10_df_dict (dict): A dictionary where keys are class indices and values are ndarrays of the top top_n SHAP values (floats) for each class.
    """
    top_10_features_dict = {}
    top_10_df_dict = {}
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    if len(shap_values.shape) < 3:
        class_shap_values = mean_abs_shap_values[:]
        top_indices = np.argsort(class_shap_values)[-top_n:][::-1]
        top_features = [str(f) for f in np.array(feature_names)[top_indices]]
        top_values = class_shap_values[top_indices]
        top_10_features_dict[1] = top_features
        top_10_df_dict[1] = top_values
    if len(shap_values.shape) == 3:
        num = (mean_abs_shap_values.shape[1])
        for class_idx in range(num):
            class_shap_values = mean_abs_shap_values[:, class_idx]
            top_indices = np.argsort(class_shap_values)[-top_n:][::-1]
            top_features = [str(f) for f in np.array(feature_names)[top_indices]]
            top_values = class_shap_values[top_indices]
            top_10_features_dict[class_idx] = top_features
            top_10_df_dict[class_idx] = top_values

    return top_10_features_dict, top_10_df_dict


def plot_shap_top_features_per_class(top_10_features_dict, top_10_df_dict, pdf):
    """
    This function generates horizontal bar plots of the top SHAP feature values for each class in a classification task.

    Parameters
    - top_10_features_dict (dict): A dictionary where each key represents a class label and the corresponding value is a list of the top feature names for that class.
    - top_10_df_dict (dict): A dictionary where each key corresponds to a class label, and the value is a ndarray of SHAP values for the top features of that class.
    - le (LabelEncoder):A label encoder instance used to map class labels back to their original class names.
    - pdf (matplotlib.backends.backend_pdf.PdfPages, optional): An open PdfPages object where plots will be saved. If None, plots will only be displayed on the screen and not saved.

    Returns
    None

    Author: Sri
    """
    for i in top_10_features_dict:
        top_features = [str(f) if isinstance(f, (float, int)) else f for f in top_10_features_dict[i]]
        top_values = top_10_df_dict[i]
        plt.figure(figsize=(8, 6))
        plt.barh(top_features, top_values)
        plt.xlabel('Mean SHAP Value')
        plt.ylabel('Feature')
        plt.title(f'Top {len(top_10_features_dict[i])} Features for Class {constants.VOC_REVERSE_DICT[i]}')
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        commons.save_in_pdf(pdf)


def get_class_all_data(x_train, y_train, top_10_features_dict):
    """
    The get_class_all_data function organizes training data into subsets based on the top features for each class.
    Parameters
    - x_train (pd.DataFrame): The training dataset containing feature values for all samples.
    - y_train (pd.DataFrame or pd.Series): The training labels corresponding to X_train. These labels are used to identify the class of each sample.
    - top_10_features_dict (dict): A dictionary mapping each class label to a list of its top 10 most important features.

    Returns
    - class_data (dict): A dictionary where each key is a class label, and each value is a DataFrame containing the subset of X_train rows belonging to that class, restricted to the corresponding top features.
    - all_data (dict): A dictionary where each key is a class label, and each value is a DataFrame containing the full X_train dataset, restricted to the top features for that class.

    Author: Sri
    """
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train, columns=[f"Feature_{i}" for i in range(len(x_train[0]))])

    if isinstance(y_train, pd.DataFrame):
        if len(y_train.columns) == 1:
            y_train = y_train.rename(columns={y_train.columns[0]: "target"})
        else:
            raise ValueError("y_train must have exactly one column representing the target.")
    elif isinstance(y_train, pd.Series):
        y_train = y_train.to_frame(name="target")
    elif isinstance(y_train, (list, np.ndarray)):
        y_train = pd.DataFrame(y_train, columns=['target'])
    else:
        raise ValueError(f"y_train must be a DataFrame,Series,list,array. it is a {type(y_train)}")

    class_data, all_data = {}, {}

    for class_label, top_features in top_10_features_dict.items():
        if not top_features:
            print(f"No top features found for class {class_label}. Skipping.")
            continue

        subset = x_train[y_train['target'] == class_label].loc[:, top_features].reset_index(drop=True)
        if not subset.empty:
            class_data[class_label] = subset

        if y_train['target'].isin([class_label]).any():
            all_data_subset = x_train.loc[:, top_features].reset_index(drop=True)
            if not all_data_subset.empty:
                all_data[class_label] = all_data_subset

    return class_data, all_data


from sklearn.manifold import TSNE
def fit_tsne(class_data, all_data, perplexity=30):
    """
    The fit_tsne function applies the t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm to reduce the dimensionality of datasets for each class in the provided dictionaries.

    Parameters
    - class_data (dict): A dictionary where each key is a class label, and each value is a DataFrame containing samples belonging to that class, restricted to the relevant features.
    - all_data (dict): A dictionary where each key is a class label, and each value is a DataFrame containing all samples in the dataset, restricted to the top features for the corresponding class.
    - perplexity (int, optional, default=30): The t-SNE perplexity parameter, which balances the number of nearest neighbors considered during dimensionality reduction.

    Returns
    - tsne_class (dict): A dictionary where each key is a class label, and each value is a 2D array of t-SNE-transformed data for the class-specific samples in class_data.
    - tsne_all (dict): A dictionary where each key is a class label, and each value is a 2D array of t-SNE-transformed data for all samples in all_data, restricted to the top features for the class.

    Author: Sri
    """
    tsne_class, tsne_all = {}, {}

    for class_label in class_data.keys():
        try:
            if class_data[class_label].empty or all_data[class_label].empty:
                print(f"Empty data for class {class_label}. Skipping.")
                continue

            # Adjust perplexity
            effective_perplexity = min(perplexity, len(class_data[class_label]) - 1)
            if effective_perplexity < 5:
                print(f"Not enough samples for class {class_label} to run t-SNE. Skipping.")
                continue

            tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42)

            # Validate numeric data
            if not all(np.issubdtype(dt, np.number) for dt in class_data[class_label].dtypes):
                return {}, {}

            tsne_class[class_label] = tsne.fit_transform(class_data[class_label])
            tsne_all[class_label] = tsne.fit_transform(all_data[class_label])
        except ValueError as e:
            print("ValueError for class")
        except Exception as e:
            print(f"Unexpected error for class {class_label}: {e}")

    return tsne_class, tsne_all

import matplotlib.pyplot as plt
def plot_tsne_per_class(tsne_class, tsne_all, pdf):
    """
    The plot_tsne function visualizes the results of t-SNE embeddings for each class.

    Parameters
    - tsne_class (dict): A dictionary where each key is a class label, and each value is a 2D array representing the t-SNE-transformed data for class-specific samples.
    - tsne_all (dict): A dictionary where each key is a class label, and each value is a 2D array representing the t-SNE-transformed data for all samples restricted to the top features of the corresponding class.
    - pdf (PdfPages or None, optional): An optional PdfPages object for saving plots. If provided, each plot is saved to the PDF file. If None, the plots are displayed interactively.

    Returns:
    None

    Author: Sri
    """
    # Plot the results
    for class_label in tsne_class.keys():
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_all[class_label][:, 0], tsne_all[class_label][:, 1], alpha=0.5, label="All Data", color="gray")
        plt.scatter(tsne_class[class_label][:, 0], tsne_class[class_label][:, 1], alpha=0.8,
                    label=f"Class {class_label}")
        plt.title(f't-SNE for Class {class_label}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        commons.save_in_pdf(pdf)









# from scipy.signal import savgol_filter  
# smoothed_intensity = savgol_filter(column_averages, window_length=5, polyorder=2)  # Adjust window_length and polyorder as needed

