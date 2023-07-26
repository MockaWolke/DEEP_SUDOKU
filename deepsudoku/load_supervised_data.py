import tensorflow as tf
import os 
import numpy as np
import pandas as pd
from typing import NamedTuple, Dict
from deepsudoku import REPO_PATH
from collections import namedtuple


def get_data(kind : str):
    """Get Data Arrays

    Args:
        kind (str): which data

    Returns:
        NamedTuple[np.ndarray, np.ndarray, pd.DataFrame]: _description_
    """
    
    df = pd.read_feather(REPO_PATH / "datasets/info.feather")
    
    assert kind in df.data_type.unique(), f"{kind} not in {df.data_type.unique()}"
    
    mask = kind == df.data_type
    
    inputs = np.load(REPO_PATH / "datasets/all_inputs.npy")[mask]
    labels = np.load(REPO_PATH / "datasets/all_labels.npy")[mask]
    
    df = df.loc[mask]
    
    return_tuple = namedtuple(f"{kind}_data", ["inputs","labels","df"])
    
    return return_tuple(inputs, labels, df)

def get_data_by_difficulty(kind: str) -> Dict[np.array, np.array]:
    
    inputs, labels, df = get_data(kind)
    
    return_dict = {}
    
    for diff in df.difficulty.unique():
        
        mask = df.difficulty == diff
        
        return_dict[diff] = (inputs[mask], labels[mask])
    
    return return_dict
    
def get_tf_dataset(inputs, labels):
    
    ds = tf.data.Dataset.from_tensor_slices((inputs,labels))
    
    ds = ds.map(lambda x,y : (tf.cast(tf.one_hot(x, 10, axis = -1), tf.float32),y - 1))
    
    return ds