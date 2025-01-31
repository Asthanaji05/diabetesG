import os
import pandas as pd
def get_file_name(file_path):
    """
    Function: get_file_name
    Parameters:
        file_path (str): The path to the file.
    Return:
        str: The name of the file with its extension.
    Description:
        This function takes a file path as input and returns the name of the file with its extension.
    Author:
        Shashank Asthana , Copilot
    """
    return os.path.basename(file_path)

def get_file_base_name(file_path):
    """
        Function: get_file_base_name
        Parameters:
            file_path (str): The path to the file.
        Return:
            str: The base name of the file without its extension.
        Description:
            This function takes a file path as input and returns the base name of the file without its extension.
        Author:
            Shashank Asthana , Copilot
    """
    return os.path.splitext(get_file_name(file_path))[0]

def get_file_dir(file_path):
    """
    Function: get_file_dir
    Parameters:
        file_path (str): The path to the file.
    Return:
        str: The directory of the file.
    Description:
        This function takes a file path as input and returns the directory of the file.
    Author:
        Shashank Asthana , Copilot
    """
    return os.path.dirname(file_path)


def read_csv(data_dir, data_file):
    """
    Function: read_csv
    Parameters:
        data_dir (str): The directory where the CSV file is located.
        data_file (str): The name of the CSV file.
    Return:
        DataFrame: The DataFrame containing the data from the CSV file.
    Description:
        This function reads the CSV file located in the specified directory and returns a DataFrame containing the data.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(data_dir, data_file))
    return df

def has_fractional_values(predictions):
    """
    Function: has_fractional_values
    Parameters:
        predictions (list): A list of prediction values.
    Return:
        bool: True if there are fractional values in the list, False otherwise.
    Description:
        This function checks if there are any fractional values in the list of predictions.
    """
    return any(value % 1 != 0 for value in predictions)

def round_fractional_predictions(predictions):
    if has_fractional_values(predictions):
        return [round(value) for value in predictions]
    return predictions

def save_params_as_json(model_file_path, model_params):
    import json
    with open(model_file_path, 'w') as f:
        json.dump(model_params, f, indent=4)


import os

def create_file_path(directory, file_name):
    """
    Combines the given directory and file name to create a file path.

    Args:
    - directory (str): The directory where the file should be saved.
    - file_name (str): The name of the file to be saved.

    Returns:
    - str: The full file path.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Join the directory and file name to create the full file path
    file_path = os.path.join(directory, file_name)
    
    return file_path


import pandas as pd
# commons.save_list_as_csv(constants.RESULTS_DIR, "hypertuningResults.csv", results)
import os
import pandas as pd

def save_list_as_csv(filepath, filename, data, header=False):
    """
    Saves a list of lists (or any iterable) as a CSV file using pandas.

    Args:
        filepath (str): The directory path where the CSV file will be saved.
        filename (str): The name of the CSV file to save the data to.
        data (list): A list of lists (or any iterable) where each inner list represents a row.
        header (bool, optional): If True, includes column headers. Default is False.

    Returns:
        None
    """
    try:
        # Convert the list to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Write to CSV
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        df.to_csv(os.path.join(filepath, filename), index=False, header=header)
        print(f"Data has been successfully saved to {os.path.join(filepath, filename)}")
    
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

import ML.constants as constants
from sklearn.decomposition import PCA,IncrementalPCA
import numpy as np
def load_json_model(model_path, binary_flag=False):
    """
    Loads a JSON-based machine learning model from the specified file path.

    Parameters:
    - model_path (str): The file path of the JSON model file.
    - binary_flag (bool): A flag used for specific model types (e.g., binary classification). Default is False.

    Returns:
    - object: The loaded model object.

    Author: Ritu, Sri
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model_file_name = get_file_name(model_path)
        model_name = model_file_name.strip(".json")
        index = model_name.rfind('_')
        if index > 0:
            model_name = model_name[:index]
        index = model_name.find('conc')
        if index > 0:
            model_name = model_name[index + 5:]
        index = model_name.find('_')
        if index > 0:
            model_name = model_name[:index]

        index = model_name.find('_')
        if index > 0:
            model_name = model_name[:index]
        loaded_model = None
        if model_name == "cat":
            loaded_model = constants.BEST_MODELS_DICT[model_name]
        #elif model_name == "xgbc":
        #    loaded_model = constants.BEST_MODELS_DICT[model_name][binary_flag]
        #else:
        #    loaded_model = constants.BEST_MODELS_DICT[model_name]
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        loaded_model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return loaded_model
    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")
    
import joblib
def pca(x, pca_filename,pca_fit,incremental_pca=False, k=10):
    """
    The pca function performs Principal Component Analysis (PCA) on a given dataset to reduce its dimensionality

    Parameters
    x (array-like or DataFrame): The input data for PCA. It should be in the form of a 2D array or a Pandas DataFrame, where rows represent samples and columns represent features.
    filename (str): The name (without extension) to save the trained PCA model as a .joblib file. This file can later be used to apply the same PCA transformation to new data.
    K (int, optional): The number of principal components to retain. Defaults to 10.

    Returns
    data_pca (Pandas DataFrame): A DataFrame containing the transformed data in the reduced dimensional space. The columns are labeled as PC0, PC1, ..., PC(K-1) corresponding to the principal components.

    Author: Sri (sri@vionix.bio)
    """
    if k <= 0:
        raise ValueError("Number of principal components must be greater than 0.")
    if isinstance(x, list) and all(isinstance(i, list) for i in x):
    # Convert list of lists to DataFrame
        x = pd.DataFrame(x)
    if x.empty:
        raise ValueError("empty dataframe is given")
    if x.isnull().values.any():
        raise ValueError("preprocess the null values")
    k = min(k, x.shape[1])
    if pca_fit==True and incremental_pca==False:  
        ipca = IncrementalPCA(n_components=k)
        ipca.partial_fit(x)
        joblib.dump(ipca, f"{constants.PCA_DIR}\\pca_{pca_filename}.pkl")
        print("PCA initiated and fitted then saved successfullyy")
    elif pca_fit==False and incremental_pca==True:
        ipca = joblib.load(f"{constants.PCA_DIR}\\pca_{pca_filename}.pkl")
        ipca.partial_fit(x)
        joblib.dump(ipca, f"{constants.PCA_DIR}\\pca_{pca_filename}.pkl")
        print("PCA loaded and partially fitted then saved successfully")
    elif pca_fit==False and incremental_pca==False:
        ipca = joblib.load(f"{constants.PCA_DIR}\\pca_{pca_filename}.pkl")
        print("trained pca saved successfully")
    x_pca = ipca.transform(x)
    data_pca = pd.DataFrame(x_pca, columns=[i for i in range(k)])
    return data_pca

import json
def save_load_pca(filename,ipca=None):
    """ 
    This function facilitates the saving and loading of an IncrementalPCA model's attributes to/from a JSON file.

    Parameters:
    - filename (str):The name of the file to save to or load from. This should include the full path if the file is not in the current working directory.
    - ipca (IncrementalPCA or None): If an IncrementalPCA object is provided, the function extracts its attributes and saves them to the specified file in JSON format. If None, the function reads the file specified by filename and loads the stored attributes.
    
    Returns:
    When ipca is provided (ipca != None):
    A dictionary containing the attributes of the provided IncrementalPCA object. 
    When ipca is None:
    A dictionary loaded from the specified file, containing the attributes.
    """
    if ipca != None:
        ipca_details = {
            "components": ipca.components_.tolist(),
            "mean": ipca.mean_.tolist(),
            "explained_variance": ipca.explained_variance_.tolist(),
            "explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
            "singular_values": ipca.singular_values_.tolist(),
            "n_samples_seen": int(ipca.n_samples_seen_),
        }
        with open(filename, 'w') as json_file:
            json.dump(ipca_details, json_file)
        return ipca_details
    else:
        with open(filename, 'r') as json_file:
            ipca_details = json.load(json_file)
        return ipca_details

def json_pca(x, pca_filename,pca_fit,incremental_pca=False, k=10):
    """ 
    This function performs Principal Component Analysis (PCA) or Incremental PCA (IPCA) on a dataset and manages PCA models by saving or loading their state in/from a JSON file. It supports fitting new PCA models, incrementally updating existing PCA models, or applying previously saved PCA models to new data.

    Parameters:
    - x (list or DataFrame): The input data for PCA. It can be a list of lists or a pandas DataFrame. Each row represents a sample, and each column represents a feature.
    - pca_filename (str): The name of the file (excluding extension) to save or load the PCA model. The file will be saved with a .json extension.
    - pca_fit (bool): Specifies whether to fit a new PCA model (True) or load an existing one (False).
    - incremental_pca (bool): Specifies whether to use Incremental PCA.
    - k (int, default=10): The number of principal components to retain.

    Returns:
    - data_pca (DataFrame): A pandas DataFrame containing the PCA-transformed data with columns named as integers (0, 1, ..., k-1).
    
    Note: The data must not contain null values, and preprocessing is required to handle them before calling the function.

"""
    if k <= 0:
        raise ValueError("Number of principal components must be greater than 0.")
    if isinstance(x, list) and all(isinstance(i, list) for i in x):
        x = pd.DataFrame(x)
    if x.empty:
        raise ValueError("empty dataframe is given")
    if x.isnull().values.any():
        raise ValueError("preprocess the null values")
    k = min(k, x.shape[1])
    filename=f"{constants.PCA_DIR}\\pca_{pca_filename}.json"  
    if pca_fit==True and incremental_pca==False:  
        #fitting for new data 
        ipca = IncrementalPCA(n_components=k)
        ipca.partial_fit(x)
        ipca_details=save_load_pca(filename,ipca)
        print("PCA details saved as json file")
    elif pca_fit==False and incremental_pca==True:
        #incrementally training for unseen data , note that the incremental training sample size should always be greater than the n_component size
        ipca_state = save_load_pca(filename)
        ipca = IncrementalPCA(n_components=len(ipca_state["components"]))
        ipca.components_ = np.array(ipca_state["components"])
        ipca.mean_ = np.array(ipca_state["mean"])
        ipca.explained_variance_ = np.array(ipca_state["explained_variance"])
        ipca.explained_variance_ratio_ = np.array(ipca_state["explained_variance_ratio"])
        ipca.singular_values_ = np.array(ipca_state["singular_values"])
        ipca.n_samples_seen_ = ipca_state["n_samples_seen"]
        ipca.var_ = np.zeros(ipca.mean_.shape)
        ipca.partial_fit(x)
        ipca_details=save_load_pca(filename,ipca)
        print("Incremental PCA details saved as json file")
        k = len(ipca.components_) 
    elif pca_fit==False and incremental_pca==False:
        ipca_details=save_load_pca(filename)
        print("loaded the json pca file successfully")
        k = len(ipca_details["components"])
    components = np.array(ipca_details["components"])
    mean = np.array(ipca_details["mean"])
    x_pca = np.dot(x - mean, components.T)
    data_pca = pd.DataFrame(x_pca, columns=[i for i in range(k)])
    return data_pca


def get_model_name_prefix(conc_flag, area_flag, voc_name = None):
    """
    Generates a prefix for a model name based on specified flags and optional VOC name.

    Args:
        conc_flag (bool): If True, indicates the model is concentration-specific and appends the `voc_name` to the prefix.
                              Assumes `voc_name` is provided when this flag is set.
        area_flag (bool): If True, indicates the model is area-specific and prepends 'area_based_' to the prefix.
        voc_name (str, optional): The VOC (Volatile Organic Compound) name used when `conc_flag` is True. Defaults to None.

    Returns:
        str: A prefix for the model name. The prefix format varies depending on the flag combinations:
                - `'voc_name_conc_'` for concentration-specific models.
                - `'area_based_'` for area-specific models.
                - `'area_based_voc_name_conc_'` if both flags are True.

    Raises:
        ValueError: If `conc_flag` is True but `voc_name` is not provided or is empty.
        TypeError: If `conc_flag`, `area_flag`, or `voc_name` is of an incorrect type.
    Example Usage:
        >>> get_model_name_prefix(conc_flag=True, area_flag=False, voc_name='benzene')
            'benzene_conc_'

        >>> get_model_name_prefix(conc_flag=False, area_flag=True)
            'area_based_'

        >>> get_model_name_prefix(conc_flag=True, area_flag=True, voc_name='toluene')
            'area_based_toluene_conc_'

        >>> get_model_name_prefix(conc_flag=False, area_flag=False)
            ''

    Author: Ritu
    """
    # Check that conc_flag and area_flag are boolean values
    if not isinstance(conc_flag, bool):
        raise TypeError("conc_flag must be a boolean value.")

    if not isinstance(area_flag, bool):
        raise TypeError("area_flag must be a boolean value.")

    # Check that voc_name, if provided, is a valid string
    if voc_name is not None and not isinstance(voc_name, str):
        raise TypeError("voc_name must be a string.")

    # Ensure voc_name is provided if conc_flag is True
    if conc_flag and not voc_name:
        raise ValueError("voc_name must be provided when conc_flag is True.")

    # Ensure voc_name is not an empty string
    if voc_name == '':
        raise ValueError("voc_name cannot be an empty string.")

    model_name_prefix = ''
    if conc_flag: #voc_name will always be there if conc_flag is set
        model_name_prefix = voc_name + '_conc_'
    if area_flag:
        model_name_prefix = 'area_based_' + model_name_prefix
    return model_name_prefix


