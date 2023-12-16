import pandas as pd

def read_csv_file(file_path: str, sep: str = ';', file_encoding: str = 'latin1') -> pd.DataFrame:
    """
    Reads a CSV file and returns it as a pandas DataFrame.

    :param file_path: The path to the CSV file.
    :param sep: The separator used in the CSV file (default is ';').
    :param file_encoding: The encoding of the file (default is 'latin1').
    :return: A pandas DataFrame containing the data from the CSV file.
    """
    try:
        data_df = pd.read_csv(file_path, sep=sep, encoding=file_encoding)
        return data_df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
