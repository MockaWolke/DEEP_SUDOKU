
import tensorflow as tf
import numpy as np
from architectures import MODELS

count_params = lambda model:  sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


DEVICE = "GPU:0"

with tf.device(DEVICE):

    for name, model in MODELS.items():
        
        
        try:
            model= model()
            
            
            model.build(input_shape=(None, 9, 9, 10))
        
        except:
            model= model()
                
                
            model.build(input_shape=(None, 9* 9* 10))
                
        c = count_params(model) 
            
        
        print(f"{name}: {c}")
    