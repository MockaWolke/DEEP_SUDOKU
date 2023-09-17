import tensorflow as tf
from load_data import get_tf_dataset
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class SudokuDoubleSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SudokuDoubleSoftmaxLayer, self).__init__(**kwargs)

    def call(self, x,):
        return self.sudoku_double_softmax(x)

    def sudoku_double_softmax(self, x):

        maximum = tf.reduce_max(x, axis = [1,2], keepdims= True)
        
        exp = tf.exp(x - maximum)
        
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

            
        log_sum_counter = tf.math.log( sum_counter + 1e-8)

        x = x - maximum - log_sum_counter

        x = tf.nn.softmax(x, -1)

        return x


class SudukoWrapper(tf.keras.Model):
    def __init__(self, back_bone):
        super().__init__()

        self.back_bone = back_bone

        self.metric_list = [
            tf.keras.metrics.Mean("loss"),
            tf.keras.metrics.Accuracy("accuracy"),
            SudokuWinRate("winrate"),
        ]

    def test_back_bone(self):
        x = tf.random.uniform(
            (10, 9, 9, 10),
            0,
            1,
        )

        y = self.back_bone(x)

        shape = tuple(tf.shape(y).numpy())

        assert shape == (10, 9, 9, 9), f"Wrong Shape: {shape}"

    def call(self, x, training=False):
        x = tf.cast(tf.one_hot(x, 10), tf.float32)

        return self.back_bone(x, training=training)

    @tf.function
    def train_step(self, data):
        x, y = data
        y_minus_1_for_cross_entropy = y - 1

        with tf.GradientTape() as tape:
            y_pred = self.call(x)

            loss = self.compute_loss(y=y_minus_1_for_cross_entropy, y_pred=y_pred)

        if tf.math.is_nan(loss):

            tf.print("\n\n------------------------------- Loss NAN -------------------------------\n\n")
            tf.print("x :", x)
            tf.print("y :", y)
            tf.print("y_pred :", tf.math.reduce_any(tf.math.is_nan(y_pred)))
            tf.print("\n\n-------------------------------  -------------------------------")
            


    
        
        gradients = tape.gradient(loss, self.back_bone.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.back_bone.trainable_variables)
        )

        y_pred_argmaxed = tf.argmax(y_pred, -1) + 1

        self.metric_list[0].update_state(loss)

        for metric in self.metric_list[1:]:
            if "masked" in metric.name:
                metric.update_state(y, y_pred_argmaxed, x)

            else:
                metric.update_state(y, y_pred_argmaxed)

        return {m.name: m.result() for m in self.metric_list}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_minus_1_for_cross_entropy = y - 1

        y_pred = self.call(x, training=False)

        loss = self.compute_loss(y=y_minus_1_for_cross_entropy, y_pred=y_pred)

        if tf.math.is_nan(loss):

            tf.print("\n\n------------------------------- Loss NAN -------------------------------")
            tf.print(x)
            tf.print(y)
            tf.print(y_pred)
            tf.print("\n\n-------------------------------  -------------------------------")

        else:

            y_pred_argmaxed = tf.argmax(y_pred, -1) + 1

            self.metric_list[0].update_state(loss)

            for metric in self.metric_list[1:]:
                if "masked" in metric.name:
                    metric.update_state(y, y_pred_argmaxed, x)

                else:
                    metric.update_state(y, y_pred_argmaxed)

        return {m.name: m.result() for m in self.metric_list}


class ValidationMetricsDifficulty(tf.keras.callbacks.Callback):
    def __init__(
        self,
        validation_data_by_difficulty,
        batchsize,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    ):
        super().__init__()
        self.validation_data_by_difficulty = validation_data_by_difficulty
        self.loss = loss

        self.preprocessed = {}

        for difficulty, data in self.validation_data_by_difficulty.items():
            self.preprocessed[difficulty] = get_tf_dataset(data).batch(batchsize)

        self.all_datapoints = sum(
            [len(x) for x in self.validation_data_by_difficulty.values()]
        )

        self.weight = {
            difficulty: len(data) / self.all_datapoints
            for difficulty, data in self.validation_data_by_difficulty.items()
        }

        self.winrate = SudokuWinRate()
        self.acc = tf.keras.metrics.Accuracy()
        self.loss_metric = tf.keras.metrics.Mean()

    @tf.function
    def calculate_all_metrics(self):
        results = {
            "val_sudoku_win_rate": tf.convert_to_tensor(0.0),
            "val_loss": tf.convert_to_tensor(0.0),
            "val_accuracy": tf.convert_to_tensor(0.0),
        }

        for difficulty, ds in self.preprocessed.items():
            for x, y in ds:
                pred = self.model(x, training=False)
                loss = self.loss(y - 1, pred)
                self.loss_metric.update_state(loss)

                pred = tf.argmax(pred, axis=-1) + 1

                self.acc.update_state(y, pred)
                self.winrate.update_state(y, pred)

            sudoku_win_rate = self.winrate.result()
            loss = self.loss_metric.result()
            accuracy = self.acc.result()

            results[f"val_loss/{difficulty}"] = loss
            results[f"val_sudoku_win_rate/{difficulty}"] = sudoku_win_rate
            results[f"val_accuracy/{difficulty}"] = accuracy

            results["val_loss"] += self.weight[difficulty] * loss
            results["val_sudoku_win_rate"] += self.weight[difficulty] * sudoku_win_rate
            results["val_accuracy"] += self.weight[difficulty] * accuracy

            self.winrate.reset_states()
            self.acc.reset_states()
            self.loss_metric.reset_states()

        return results

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get the memory info before calculation
        memory_info_before = tf.config.experimental.get_memory_info("GPU:0")
        max_memory_before = memory_info_before["peak"]

        # Log the maximum memory before calculation
        logger.info(
            f"Maximum VRAM memory consumption before calculation: {max_memory_before}"
        )
        print(
            f"Maximum VRAM memory consumption before calculation: {max_memory_before}"
        )

        # Perform the calculation
        results = self.calculate_all_metrics()

        # Get the memory info after calculation
        memory_info_after = tf.config.experimental.get_memory_info("GPU:0")
        max_memory_after = memory_info_after["peak"]

        # Log the maximum memory after calculation
        logger.info(
            f"Maximum VRAM memory consumption after calculation: {max_memory_after}"
        )
        print(f"Maximum VRAM memory consumption after calculation: {max_memory_after}")

        for i, a in results.items():
            logs[i] = a.numpy()
