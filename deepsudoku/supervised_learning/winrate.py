import tensorflow as tf

class SudokuWinRate(tf.keras.metrics.Metric):
    def __init__(self, name="sudoku_win_rate", **kwargs):
        super(SudokuWinRate, self).__init__(name=name, **kwargs)
        self.total_wins = self.add_weight(name="total_wins", initializer="zeros")
        self.total_puzzles = self.add_weight(name="total_puzzles", initializer="zeros")

    def update_state(self, y_true, y_pred_argmaxed, sample_weight=None):
        y_pred_argmaxed = tf.cast(y_pred_argmaxed, y_true.dtype)
        wins = tf.reduce_all(tf.equal(y_true, y_pred_argmaxed), axis=[1, 2])

        self.total_wins.assign_add(tf.reduce_sum(tf.cast(wins, tf.float32)))
        self.total_puzzles.assign_add(tf.cast(tf.size(wins), tf.float32))

    def result(self):
        return self.total_wins / self.total_puzzles
