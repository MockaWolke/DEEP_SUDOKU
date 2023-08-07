import tensorflow as tf
from deepsudoku.supervised_learning.load_supervised_data import get_tf_dataset
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        
class ValidationMetricsDifficulty(tf.keras.callbacks.Callback):
    def __init__(self, validation_data_by_difficulty, batchsize,  loss = tf.keras.losses.SparseCategoricalCrossentropy()):
        super().__init__()
        self.validation_data_by_difficulty = validation_data_by_difficulty
        self.loss = loss
        
        self.preprocessed = {}
        
        for difficulty, data in self.validation_data_by_difficulty.items():
            self.preprocessed[difficulty] = get_tf_dataset(data).batch(batchsize)
        
        self.all_datapoints = sum([len(x) for x in self.validation_data_by_difficulty.values()])
        
        self.weight = {difficulty: len(data)/self.all_datapoints for difficulty, data in self.validation_data_by_difficulty.items()}
        
        self.winrate = SudokuWinRate()
        self.acc = tf.keras.metrics.Accuracy()
        self.loss_metric = tf.keras.metrics.Mean()

    @tf.function
    def calculate_all_metrics(self):
        
        results = {"val_sudoku_win_rate":tf.convert_to_tensor(0.0),
                   "val_loss":tf.convert_to_tensor(0.0),
                   "val_accuracy":tf.convert_to_tensor(0.0)}
        
        for difficulty, ds in self.preprocessed.items():
        
            for x,y in ds:
                
                pred = self.model(x, training = False)
                loss = self.loss(y, pred)
                self.loss_metric.update_state(loss)
                self.acc.update_state(y, tf.argmax(pred, axis = -1))
                self.winrate.update_state(y, pred)
                
            sudoku_win_rate =  self.winrate.result()
            loss =  self.loss_metric.result()
            accuracy =  self.acc.result()
                
            results[f"val_sudoku_win_rate/{difficulty}"] = sudoku_win_rate
            results[f"val_loss/{difficulty}"] = loss
            results[f"val_accuracy/{difficulty}"] = accuracy
            
            
            results["val_sudoku_win_rate"] += self.weight[difficulty] * sudoku_win_rate
            results["val_loss"] += self.weight[difficulty] * loss
            results["val_accuracy"] += self.weight[difficulty] * accuracy
    
            self.winrate.reset_states()
            self.acc.reset_states()
            self.loss_metric.reset_states()
    
        return results
                
            
        

    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
    
        # Get the memory info before calculation
        memory_info_before = tf.config.experimental.get_memory_info('GPU:0')
        max_memory_before = memory_info_before['peak']

        # Log the maximum memory before calculation
        logger.info(f"Maximum VRAM memory consumption before calculation: {max_memory_before}")
        print(f"Maximum VRAM memory consumption before calculation: {max_memory_before}")

        # Perform the calculation
        results = self.calculate_all_metrics()

        # Get the memory info after calculation
        memory_info_after = tf.config.experimental.get_memory_info('GPU:0')
        max_memory_after = memory_info_after['peak']

        # Log the maximum memory after calculation
        logger.info(f"Maximum VRAM memory consumption after calculation: {max_memory_after}")
        print(f"Maximum VRAM memory consumption after calculation: {max_memory_after}")

        for i,a in results.items():
            logs[i] = a.numpy()
