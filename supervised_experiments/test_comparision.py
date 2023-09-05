import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Flatten, Dense, Reshape, Softmax
from keras.utils.layer_utils import count_params  
from tensorflow.keras import layers
import numpy as np
from metrics_and_wrapper import SudokuWinRate, MaskedAccuracy, MaskedSudokuWinRate, SudukoWrapper, ValidationMetricsDifficulty
from architectures import *
from load_data import get_data, get_tf_dataset, get_data_by_difficulty_and_origin

DEVICE = "GPU:0"

with tf.device(DEVICE):
    
    print("Test Load Data")
    
    train_df = get_data("train", "comp_old_data").sample(frac=1)
    
    print(f"\n\nThe training length: {len(train_df)}\n\n")
    
    train_ds = get_tf_dataset(train_df)
    train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    validation_seperated = get_data_by_difficulty_and_origin("val", "comp_old_data")

    callback = ValidationMetricsDifficulty(validation_seperated, 32)
    
    for x,y in train_ds.take(1):
        
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

    callback.set_model(model)

    logs = {"smth":3}
    

    callback.on_epoch_end(0, logs)
    
    print(logs)
    for name, val in logs.items():
        
        if np.isnan(val):
            
            print(f"\n{name} is NAN!!")
    