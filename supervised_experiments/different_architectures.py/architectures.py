import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Flatten, Dense, Reshape, Softmax
from keras.utils.layer_utils import count_params  
from tensorflow.keras import layers
import numpy as np
from metrics_and_wrapper import SudokuWinRate, MaskedAccuracy, MaskedSudokuWinRate, SudukoWrapper



def same_conv_with_fc_head():
    
    input_tensor = tf.keras.Input(shape=(9, 9, 10))

    # Network layers
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, kernel_size=9, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # Actor layer
    x = Dense(9**3)(x)
    x = tf.keras.layers.Reshape((9,9,9))(x)
    output_tensor = tf.keras.layers.Softmax()(x)
    # Create the Model
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
def same_conv_conv_head():
    
    input_tensor = tf.keras.Input(shape=(9, 9, 10))

    # Network layers
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, kernel_size=9, padding='same', activation='relu')(x)
    output_tensor = Conv2D(9,kernel_size=9,padding = "same", activation = "softmax")(x)

    # Create the Model
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

def same_fc_mlp():
    
    input_tensor = tf.keras.Input(shape=(9, 9, 10))

    # Flatten the input
    x = Flatten()(input_tensor)

    # Actor layers
    x = Dense(128, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    x = Dense(9**3, activation=None)(x)

    # Reshape and softmax layers
    x = Reshape((9, 9, 9))(x)
    output_tensor = Softmax()(x)

    # Create the final Model
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

def big_conv_head():
    
    model_inputs = tf.keras.Input((9,9,10))
    x = model_inputs

    x = tf.keras.layers.Conv2D(32,3, padding= "same", activation= "relu")(x)
    x = tf.keras.layers.Conv2D(64,3, padding= "same", activation= "relu")(x)
    x = tf.keras.layers.Conv2D(128,3, padding= "same", activation= "relu")(x)
    skip = x
    x = tf.keras.layers.Conv2D(128,3, padding= "same", activation= "relu")(x)
    x = tf.keras.layers.Conv2D(128,3, padding= "same", activation= "relu")(x)
    x = skip + x
    skip = x
    x = tf.keras.layers.Conv2D(128,3, padding= "same", activation= "relu")(x)
    x = tf.keras.layers.Conv2D(128,3, padding= "same", activation= "relu")(x)
    x = skip + x
    x = tf.keras.layers.Conv2D(256,9, padding= "same", activation= "relu")(x)
    x = tf.keras.layers.Conv2D(9,9, padding= "same", activation= "softmax")(x)

    return tf.keras.Model(model_inputs,x)


class BigSudokuTransformer(tf.keras.Model):
    
    def __init__(self, embed_dim=64, num_heads=4):
        
        super(BigSudokuTransformer, self).__init__()

        # Create the positional embeddings
        xs = tf.repeat(tf.range(9), 9)
        ys = tf.tile(tf.range(9), [9])

        blocks = tf.tile(tf.repeat(tf.range(3), 3), [3])
        blocks = tf.concat([blocks, blocks + 3, blocks + 6], axis=0)

        self.positional_embedding = tf.concat([
            tf.one_hot(xs, 9),
            tf.one_hot(ys, 9),
            tf.one_hot(blocks, 9),
        ], axis=-1)
        
        
        self.to_embed_dim = layers.Dense(embed_dim, activation='relu')
        self.block1 = self.get_attention_block(embed_dim, num_heads)
        self.block2 = self.get_attention_block(embed_dim, num_heads)
        self.block3 = self.get_attention_block(embed_dim, num_heads)
        self.block4 = self.get_attention_block(embed_dim, num_heads)
        self.output_layer = layers.Dense(9, activation='softmax')
        self.reshape_layer = Reshape((9,9,9))

    def get_attention_block(self, embed_dim, num_heads):
        
        inputs = tf.keras.Input((81,embed_dim))
        
        x = residual = inputs
        
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(query=x, key=x, value=x)
        
        x += residual
        
        residual = x
        x = Dense(embed_dim * 4, "relu")(x)
        x = Dense(embed_dim, "relu")(x)
        
        x += residual
            
        return tf.keras.Model(inputs, x)
    
    
    
    @tf.function
    def call(self, inputs):
        # Input Preprocessing
        obs = tf.reshape(inputs, (-1, 81, 10))
        batch_size = tf.shape(inputs)[0]
        to_batch_size = tf.tile(self.positional_embedding[tf.newaxis, :, :], [batch_size, 1, 1])
        obs = tf.concat([obs, to_batch_size], axis=-1)

        # Transformer Layers with skip connections
        x = self.to_embed_dim(obs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_layer(x)
        x = self.reshape_layer(x)
        
        return x

class SmallSudokuTransformer(tf.keras.Model):
    def __init__(self, embed_dim=64, num_heads=4):
        super(SmallSudokuTransformer, self).__init__()

        # Create the positional embeddings
        xs = tf.repeat(tf.range(9), 9)
        ys = tf.tile(tf.range(9), [9])

        blocks = tf.tile(tf.repeat(tf.range(3), 3), [3])
        blocks = tf.concat([blocks, blocks + 3, blocks + 6], axis=0)

        self.positional_embedding = tf.concat([
            tf.one_hot(xs, 9),
            tf.one_hot(ys, 9),
            tf.one_hot(blocks, 9),
        ], axis=-1)

        # Transformation Layers
        self.to_embed_dim = layers.Dense(embed_dim, activation='relu')
        self.layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.layer1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.bigger = layers.Dense(embed_dim * 2, activation='relu')
        self.smaller = layers.Dense(embed_dim, activation='relu')
        self.output_layer = layers.Dense(9, activation='softmax')
        self.reshape_layer = Reshape((9,9,9))

    @tf.function
    def call(self, inputs):
        # Input Preprocessing
        obs = tf.reshape(inputs, (-1, 81, 10))
        batch_size = tf.shape(inputs)[0]
        to_batch_size = tf.tile(self.positional_embedding[tf.newaxis, :, :], [batch_size, 1, 1])
        obs = tf.concat([obs, to_batch_size], axis=-1)

        # Transformer Layers with skip connections
        x = self.to_embed_dim(obs)

        x = self.layer(query=x, key=x, value=x)
        
        x = self.bigger(x)
        x = self.smaller(x)
        
        x = self.layer1(query=x, key=x, value=x)
        
        x = self.output_layer(x)
        x = self.reshape_layer(x)
        return x


def same_transformer():

    return SmallSudokuTransformer(128, 8)

def big_transformer():
    return BigSudokuTransformer(256,8)


MODELS = {
    "same_fc_mlp": same_fc_mlp,
    "same_conv_conv_head" : same_conv_conv_head,
    "same_conv_with_fc_head": same_conv_with_fc_head,
    "same_transformer": same_transformer,
    "big_conv_head" : big_conv_head,
    "big_transformer": big_transformer,
}

