import tensorflow as tf

class SudokuDoubleSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SudokuDoubleSoftmaxLayer, self).__init__(**kwargs)

    def call(self, x,):
        return self.sudoku_double_softmax(x)

    def sudoku_double_softmax(self, x):
        exp = tf.exp(x)

        sum_counter = tf.zeros_like(x)

        cols = tf.reduce_sum(exp, axis=2)
        rows = tf.reduce_sum(exp, axis=1)

        sum_counter += cols[:, :, None]
        sum_counter += rows[:, None]

        blocks_grouped = tf.reshape(
            tf.transpose(tf.reshape(exp, (-1, 3, 3, 3, 3, 9)), (0, 1, 3, 2, 4, 5)),
            (-1, 9, 9, 9),
        )
        blocks_sum = tf.reduce_sum(blocks_grouped, axis=2)

        sum_counter_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(sum_counter, (-1, 3, 3, 3, 3, 9)), (0, 1, 3, 2, 4, 5)
            ),
            (-1, 9, 9, 9),
        )
        sum_counter_reshaped += blocks_sum[:, :, None]
        sum_counter = tf.reshape(
            tf.transpose(
                tf.reshape(sum_counter_reshaped, (-1, 3, 3, 3, 3, 9)),
                (0, 1, 3, 2, 4, 5),
            ),
            (-1, 9, 9, 9),
        )

        x = x - tf.math.log(sum_counter)

        x = tf.nn.softmax(x, -1)

        return x