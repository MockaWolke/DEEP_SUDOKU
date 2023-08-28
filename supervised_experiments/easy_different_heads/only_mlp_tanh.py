import logging
import tensorflow as tf
from deepsudoku import REPO_PATH
import os
os.chdir(REPO_PATH)

BATCHSIZE = 256  * 2


RUNNAME = "easy_mlp"
LOG_DIR = f"logs/{RUNNAME}"
CKPT_DIR = f"ckpt/{RUNNAME}"

os.makedirs(LOG_DIR, exist_ok = True)
os.makedirs(CKPT_DIR, exist_ok = True)

logging.basicConfig(filename=os.path.join(LOG_DIR, 'logfile.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

from deepsudoku.supervised_learning import  load_supervised_data, metrics_and_callbacks

train_df = load_supervised_data.get_data("train", "easy_only").sample(frac=1)
train_ds = load_supervised_data.get_tf_dataset(train_df)
train_ds = train_ds.shuffle(1000).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

validation_seperated = load_supervised_data.get_data_by_difficulty_and_origin("val", "easy_only")



model_inputs = tf.keras.Input((9,9,10))
x = model_inputs

x = tf.keras.layers.Flatten() (x)
x = tf.keras.layers.Dense(256, activation= "tanh")(x)
x = tf.keras.layers.Dense(256 * 2, activation= "tanh")(x)
x = tf.keras.layers.Dense(256 * 4, activation= "tanh")(x)

x = tf.keras.layers.Dense(9 ** 3, activation= "linear")(x)
x = tf.keras.layers.Reshape((9,9,9))(x)
x = tf.nn.softmax(x, axis = -1)

model = tf.keras.Model(model_inputs,x)


MSE = tf.keras.losses.MeanSquaredError()
CROSS = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile("Adam", CROSS, metrics = ["accuracy", metrics_and_callbacks.SudokuWinRate()])


LOG_DIR = f"logs/{RUNNAME}"
CKPT_DIR = f"ckpt/{RUNNAME}"

os.makedirs(LOG_DIR, exist_ok = True)
os.makedirs(CKPT_DIR, exist_ok = True)

logging.basicConfig(filename=os.path.join(LOG_DIR, 'logfile.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

callbacks= [
    metrics_and_callbacks.ValidationMetricsDifficulty(validation_seperated, BATCHSIZE),
    tf.keras.callbacks.ModelCheckpoint(CKPT_DIR + "/cp-{epoch:04d}.ckpt", save_weights_only = True),
    tf.keras.callbacks.CSVLogger(LOG_DIR + "/logs.csv"),
    tf.keras.callbacks.TensorBoard(log_dir= LOG_DIR)
]


hist = model.fit(train_ds, epochs = 30, callbacks=callbacks, initial_epoch=0)


model.save(os.path.join(CKPT_DIR,"final_model"))


