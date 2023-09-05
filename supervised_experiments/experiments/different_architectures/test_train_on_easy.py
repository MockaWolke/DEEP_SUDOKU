from deepsudoku import REPO_PATH
import os

current_path = REPO_PATH / "supervised_experiments"

os.chdir(current_path)

import sys
sys.path.append(str(current_path))


import tensorflow as tf
from keras.utils.layer_utils import count_params  
import numpy as np
from deepsudoku import REPO_PATH
import os
os.chdir(REPO_PATH / "supervised_experiments")

from metrics_and_wrapper import SudokuWinRate, SudukoWrapper
from architectures import *
from load_data import get_data, get_tf_dataset

with tf.device("CPU:0"):
    
    print("Test Load Data") 
    
    train = get_data("train", "realistic_easy_data")
    
    ds = get_tf_dataset(train).batch(10)
    
    for x,y in ds:
        
        print("x:", x.shape, x.dtype)
        print("y:", y.shape, y.dtype)
        break
    
    

    print("Test Wrapper")

    back_bone = same_fc_mlp()
    
    model = SudukoWrapper(back_bone)

    model.test_back_bone()
    model.compile("Adam", tf.keras.losses.SparseCategoricalCrossentropy())
    
    
    batch = (x, y)
    
    result = model.train_step(batch)
    print(result)
    
    result = model.test_step(batch)
    print(result)


    print("------------"*5)
    print("Testing Models")
    
    for name, func in MODELS.items():
        
        print("Testing: ", name)
        
        tf.keras.backend.clear_session()
        
        model = SudukoWrapper(func())

        model.test_back_bone()
        
        print(count_params(model.trainable_variables), "\n\n")
        