import tensorflow as tf
import os 
import numpy as np
import pandas as pd
from typing import Dict
from deepsudoku import REPO_PATH


def get_data(kind : str, data_path = "all_data"):
    """Get Data Arrays

    Args:
        kind (str): which data

    Returns:
        NamedTuple[np.ndarray, np.ndarray, pd.DataFrame]: _description_
    """
    
    path = REPO_PATH / f"data/{data_path}.feather"
    
    if not os.path.exists(path):
        
        raise ValueError(f"{path} does note exists")
    
    
    
    df = pd.read_feather(path)
    
    ds_type_col = "ds_type" if "ds_type" in df.columns else "data_type"
    
    assert kind in df[ds_type_col].unique(), f"{kind} not in {df[ds_type_col].unique()}"
    
    
    if ds_type_col == "ds_type":
    
        df = df.query("ds_type == @kind")
        
    else: 
        
        df = df.query("data_type == @kind")
        
    
    return df


def get_data_by_difficulty_and_origin(kind: str, data_path = "all_data") -> Dict[np.array, np.array]:
    
    df = get_data(kind, data_path = data_path)
    
    mask = df.difficulty.notna()
    
    our_data = df.loc[mask]
    other = df.loc[~mask]
    
    return_dict = {}
        
    if len(our_data):
        
        for diff in our_data.difficulty.unique():
            
            return_dict[diff] = our_data.query("difficulty == @diff").copy()
    
    if len(other):
            
        for origin in other.origin.unique():

            return_dict[origin] = other.query("origin == @origin").copy()
        
    return return_dict


def process_data(x, y):
    # Replace '.' with '0' in the input
    x = tf.strings.regex_replace(x, "\.", "0")

    # Convert the strings to int64
    x = tf.strings.to_number(tf.strings.bytes_split(x), out_type=tf.int64)
    y = tf.strings.to_number(tf.strings.bytes_split(y), out_type=tf.int64)

    # Reshape the tensors to 9x9
    x = tf.reshape(x, (9, 9))
    y = tf.reshape(y, (9, 9))

    return x, y

    
def get_tf_dataset(df):
    
    ds = tf.data.Dataset.from_tensor_slices((df.quiz,df.solution))
    
    ds = ds.map(process_data, tf.data.AUTOTUNE)

    return ds


