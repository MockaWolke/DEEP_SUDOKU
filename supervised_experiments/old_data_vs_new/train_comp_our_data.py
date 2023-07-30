import logging
import tensorflow as tf
from deepsudoku import REPO_PATH
import os
os.chdir(REPO_PATH)

BATCHSIZE = 256 


RUNNAME = "comp_our_data"
LOG_DIR = f"logs/{RUNNAME}"
CKPT_DIR = f"ckpt/{RUNNAME}"
print(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok = True)
os.makedirs(CKPT_DIR, exist_ok = True)

logging.basicConfig(filename=os.path.join(LOG_DIR, 'logfile.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

from deepsudoku import  load_supervised_data, utils, metrics_and_callbacks

train_df = load_supervised_data.get_data("train", "comp_our_data").sample(frac=1)
train_ds = load_supervised_data.get_tf_dataset(train_df)
train_ds = train_ds.shuffle(1000).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

validation_seperated = load_supervised_data.get_data_by_difficulty_and_origin("val", "comp_our_data")


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

model = tf.keras.Model(model_inputs,x)


MSE = tf.keras.losses.MeanSquaredError()
CROSS = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile("Adam", CROSS, metrics = ["accuracy", metrics_and_callbacks.SudokuWinRate()])



callbacks= [
    metrics_and_callbacks.ValidationMetricsDifficulty(validation_seperated, BATCHSIZE),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 5),
    tf.keras.callbacks.EarlyStopping(patience = 10),
    tf.keras.callbacks.ModelCheckpoint(CKPT_DIR + "/cp-{epoch:04d}.ckpt", save_weights_only = True),
    tf.keras.callbacks.CSVLogger(LOG_DIR + "/logs.csv"),
    tf.keras.callbacks.TensorBoard(log_dir= LOG_DIR, profile_batch = '500,520')
]


hist = model.fit(train_ds, epochs=30, callbacks=callbacks, initial_epoch=0)


model.save(os.path.join(CKPT_DIR,"final_model"))


