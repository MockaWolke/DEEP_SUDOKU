import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Flatten, Dense, Reshape, Softmax
from keras.utils.layer_utils import count_params  
from tensorflow.keras import layers

class SudokuWinRate(tf.keras.metrics.Metric):
    def __init__(self, name='sudoku_win_rate', **kwargs):
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
    
class MaskedSudokuWinRate(tf.keras.metrics.Metric):
    def __init__(self, name='masked_sudoku_win_rate', **kwargs):
        super(MaskedSudokuWinRate, self).__init__(name=name, **kwargs)
        self.total_wins = self.add_weight(name="total_wins", initializer="zeros")
        self.total_puzzles = self.add_weight(name="total_puzzles", initializer="zeros")

    def update_state(self, y_true, y_pred_argmaxed, x_for_mask, sample_weight=None):

        mask = x_for_mask == 0

        is_correct = tf.equal(y_true, y_pred_argmaxed)

        masked_win = tf.logical_or(mask, is_correct)

        y_pred_argmaxed = tf.cast(y_pred_argmaxed, y_true.dtype)
        wins = tf.reduce_all(masked_win, axis=[1, 2])

        
        self.total_wins.assign_add(tf.reduce_sum(tf.cast(wins, tf.float32)))
        self.total_puzzles.assign_add(tf.cast(tf.size(wins), tf.float32))

    def result(self):
        return self.total_wins / self.total_puzzles
    
    
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        
        self.accs_sum = self.add_weight(name="accs_sum", initializer="zeros")
        self.total_puzzles = self.add_weight(name="total_puzzles", initializer="zeros")

    def update_state(self, y_true, y_pred_argmaxed, x_for_mask, sample_weight=None):

        mask = x_for_mask != 0

        empty_fields = tf.reduce_sum(tf.cast(mask, tf.float32), axis = [1, 2])
        
        y_pred_argmaxed = tf.cast(y_pred_argmaxed, y_true.dtype)
        correct = tf.equal(y_true, y_pred_argmaxed)
        
        empty_field_same = tf.logical_and(mask, correct)
        empty_field_same = tf.reduce_sum(tf.cast(empty_field_same, tf.float32), axis = [1, 2])
        
        acuracys = empty_field_same /  empty_fields 
        
        self.accs_sum.assign_add(tf.reduce_sum( acuracys))
        self.total_puzzles.assign_add(tf.cast(tf.size(acuracys), tf.float32))

    def result(self):
        return self.accs_sum / self.total_puzzles
    
    
    

    

class SudukoWrapper(tf.keras.Model):
    
    def __init__(self, back_bone):
        super().__init__()
        
        
        self.back_bone = back_bone
        
        self.metric_list = [
            tf.keras.metrics.Mean("loss"),
            tf.keras.metrics.Accuracy("accuracy"),
            MaskedAccuracy("masked_accuracy"),
            SudokuWinRate("winrate"),
            MaskedSudokuWinRate("masked_winrate"),
        ]
        
    
    def test_back_bone(self):
        
        
        x = tf.random.uniform((10, 9, 9 ,10), 0, 1,)
        
        y = self.back_bone(x)
        
        shape = tuple(tf.shape(y).numpy())
        
        assert shape == (10, 9, 9 ,9) , f"Wrong Shape: {shape}"
        
        
    def call(self, x , training = False):
        
        x = tf.cast(tf.one_hot(x, 10), tf.float32)
        
        return self.back_bone(x, training = training)
    
    @tf.function
    def train_step(self, data):
        
        x,y = data
        y_minus_1_for_cross_entropy = y - 1
        
        with tf.GradientTape() as tape:
            
            y_pred = self.call(x)
            
            loss = self.compute_loss(y=y_minus_1_for_cross_entropy, y_pred=y_pred)
            
        gradients = tape.gradient(loss, self.back_bone.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.back_bone.trainable_variables))
        
        y_pred_argmaxed = tf.argmax(y_pred, -1)
        
        self.metric_list[0].update_state(loss)
        
        for metric in self.metric_list[1:]:
            
            if "masked" in metric.name:
                
                metric.update_state(y, y_pred_argmaxed, x)
                
            else:
                
                metric.update_state(y, y_pred_argmaxed)
                
                
        return {m.name: m.result() for m in self.metric_list}    

    @tf.function
    def test_step(self, data):
        
        x,y = data
        
        y_pred = self.call(x, training = False)
        
        loss = self.compute_loss(y=y, y_pred=y_pred)
            
        y_pred_argmaxed = tf.argmax(y_pred, -1)
        
        self.metric_list[0].update_state(loss)
        
        for metric in self.metric_list[1:]:
            
            if "masked" in metric.name:
                
                metric.update_state(y, y_pred_argmaxed, x)
                
            else:
                
                metric.update_state(y, y_pred_argmaxed)
                
        return {m.name: m.result() for m in self.metric_list}   
        
    