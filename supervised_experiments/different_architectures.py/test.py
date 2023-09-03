import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Flatten, Dense, Reshape, Softmax
from keras.utils.layer_utils import count_params  
from tensorflow.keras import layers
import numpy as np
from metrics_and_wrapper import SudokuWinRate, MaskedAccuracy, MaskedSudokuWinRate, SudukoWrapper
from architectures import *


with tf.device("CPU:0"):
    
    
    
    print("Test Accuracy")
    
    
    label = np.zeros((10, 9, 9), np.int64)
    label[:, 0,0] = 1
    
    label[:2,0]  = np.arange(1, 10)
    
    pred = label.copy()
    pred[1,0] = np.arange(1, 10)[::-1]
    x = (label != 0).astype(int)
    
    label = tf.convert_to_tensor(label, tf.int64)
    pred = tf.convert_to_tensor(pred, tf.int64)
    x = tf.convert_to_tensor(x, tf.int64)
    


    acc = 0.9 + (1/9) * 0.1
    
    masked_acc = MaskedAccuracy()
    masked_acc.update_state(label, pred, x)
    result = masked_acc.result().numpy()
    
    
    assert np.allclose(acc, result), "The masked acc does not work"

    winrate = SudokuWinRate()
    winrate.update_state(label, pred)
    results = winrate.result().numpy()
    
    assert np.isclose(results, 0.9), "The winrate metric is flawed"
    
    x = x.numpy()
    
    x[1,0] = np.array([0,0,0,0,1,0,0,0,0], x.dtype)
    x = tf.convert_to_tensor(x, tf.int64)
    
    masked_winrate = MaskedSudokuWinRate()
    masked_winrate.update_state(label, pred, x)
    result = masked_winrate.result().numpy()

    assert np.isclose(result, 1.0), "masked winrate is flawed"
    

    print("Test Wrapper")

    back_bone = same_fc_mlp()
    
    model = SudukoWrapper(back_bone)

    model.test_back_bone()
    model.compile("Adam", tf.keras.losses.SparseCategoricalCrossentropy())
    
    
    batch = (x, tf.ones_like(label))
    
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
        