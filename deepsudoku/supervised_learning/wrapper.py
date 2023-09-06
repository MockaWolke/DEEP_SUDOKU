import tensorflow as tf
from deepsudoku.supervised_learning.winrate import SudokuWinRate





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

        y_pred_argmaxed = tf.argmax(y_pred, -1) + 1

        self.metric_list[0].update_state(loss)

        for metric in self.metric_list[1:]:
            if "masked" in metric.name:
                metric.update_state(y, y_pred_argmaxed, x)

            else:
                metric.update_state(y, y_pred_argmaxed)

        return {m.name: m.result() for m in self.metric_list}

    @tf.function
    def predict_step(self, data):

        if isinstance(data, tuple):

            x, _y = data
        else: 
            x = data

        y_pred = self.call(x, training=False)


        y_pred_argmaxed = tf.argmax(y_pred, -1) + 1

        return y_pred_argmaxed
