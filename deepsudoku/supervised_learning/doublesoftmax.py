import tensorflow as tf


class SudokuDoubleSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SudokuDoubleSoftmaxLayer, self).__init__(**kwargs)

    def group_blocks_to_columns(self, x):
        """Groups blocks as columns or vice versa"""
        return tf.reshape(
            tf.transpose(tf.reshape(x, (-1, 3, 3, 3, 3, 9)), (0, 1, 3, 2, 4, 5)),
            (-1, 9, 9, 9),
        )

    @tf.function
    def call(self, x):

        # suptract maximum for numerical stability
        # see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
        # because we dont really have the same maximum for everything we just take it over the whole field

        maximum = tf.reduce_max(x, axis = [1,2], keepdims= True)
        
        # make probabilities
        exp = tf.exp(x - maximum)
        
        # in this variable we will add up the values to normalize
        sum_counter = tf.zeros_like(x)

        # getting the rows and columns is trivial
        cols = tf.reduce_sum(exp, axis=2, keepdims = True)
        rows = tf.reduce_sum(exp, axis=1, keepdims = True)

        # add values
        sum_counter += cols
        sum_counter += rows

        # get blocks as columns
        blocks_grouped = self.group_blocks_to_columns(exp)
        
        # sum over blocks
        blocks_sum = tf.reduce_sum(blocks_grouped, axis=2, keepdims = True)

        sum_counter_reshaped = self.group_blocks_to_columns(sum_counter)
        sum_counter_reshaped += blocks_sum
        
        # back to normal shape
        sum_counter =  self.group_blocks_to_columns(sum_counter_reshaped)
        

        # log with small offset for numerical stability
        log_sum_counter = tf.math.log( sum_counter + 1e-8)

        # subtract the logits
        x = x - maximum - log_sum_counter

        # apply standard softmax over the last axis
        x = tf.nn.softmax(x, -1)

        return x