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

