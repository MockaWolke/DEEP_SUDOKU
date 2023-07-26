import tensorflow as tf
from deepsudoku.load_supervised_data import get_tf_dataset

class SudokuWinRate(tf.keras.metrics.Metric):
    def __init__(self, name='sudoku_win_rate', **kwargs):
        super(SudokuWinRate, self).__init__(name=name, **kwargs)
        self.total_wins = self.add_weight(name="total_wins", initializer="zeros")
        self.total_puzzles = self.add_weight(name="total_puzzles", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)
        wins = tf.reduce_all(tf.equal(y_true, y_pred), axis=[1, 2])

        
        self.total_wins.assign_add(tf.reduce_sum(tf.cast(wins, tf.float32)))
        self.total_puzzles.assign_add(tf.cast(tf.size(wins), tf.float32))

    def result(self):
        return self.total_wins / self.total_puzzles
        
        
class SudokuWinRateByDifficulty(tf.keras.callbacks.Callback):
    def __init__(self, validation_data_by_difficulty, batchsize):
        super().__init__()
        self.validation_data_by_difficulty = validation_data_by_difficulty
        
        self.preprocessed = {}
        
        for difficulty, data in self.validation_data_by_difficulty.items():
        
            self.preprocessed[difficulty] = get_tf_dataset(*data).batch(batchsize)
        
        self.metric = SudokuWinRate()

    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
                
        for difficulty, data in self.validation_data_by_difficulty.items():
            
            preds = self.model.predict(self.preprocessed[difficulty], verbose = False)
            
            labels = data[1] - 1
            
            self.metric.update_state(labels, preds)
            result = self.metric.result().numpy()
            logs[f'val_sudoku_win_rate/{difficulty}'] = result
            self.metric.reset_states()