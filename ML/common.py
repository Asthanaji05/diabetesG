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

