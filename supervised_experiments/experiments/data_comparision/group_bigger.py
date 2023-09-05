from deepsudoku import REPO_PATH
import os

current_path = REPO_PATH / "supervised_experiments"

os.chdir(current_path)

import sys
sys.path.append(str(current_path))

import sys
import logging
import tensorflow as tf
import os
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from metrics_and_wrapper import SudukoWrapper, ValidationMetricsDifficulty
from architectures import *
from load_data import get_data, get_tf_dataset, get_data_by_difficulty_and_origin
import argparse
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument("--test", type = bool, default= False)
parser.add_argument("--batchsize", type = int, default= 512)
parser.add_argument("--epochs", type = int, default= 50)
args = parser.parse_args()


args.model = "big_conv_head"
args.data_set = "comp_old_data"


start_time = time.time()

RUNNAME = f"Group_Data_bigger_{args.model}_{start_time}"


LOG_DIR = f"logs/{RUNNAME}" if not args.test else f"test_logs/{RUNNAME}"
CKPT_DIR = f"ckpt/{RUNNAME}" if not args.test else f"test_ckpt/{RUNNAME}"

print(LOG_DIR)

os.makedirs(LOG_DIR, exist_ok = True)
os.makedirs(CKPT_DIR, exist_ok = True)


back_bone = MODELS[args.model]()

model = SudukoWrapper(back_bone)


hpams_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "hpams"))

with hpams_writer.as_default():

    tf.summary.text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        0
    )
    



train_df = get_data("train", "comp_old_data").sample(frac=1)
train_ds = get_tf_dataset(train_df)
train_ds = train_ds.shuffle(1000).batch(args.batchsize).prefetch(tf.data.AUTOTUNE)

validation_seperated = get_data_by_difficulty_and_origin("val", "comp_old_data")


if args.test:
    
    train_ds = train_ds.take(2)
    args.epochs = 2

CROSS = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile("Adam", CROSS, )
logger = logging.getLogger(__name__)

callbacks= [
    ValidationMetricsDifficulty(validation_seperated, args.batchsize),
    tf.keras.callbacks.CSVLogger(LOG_DIR + "/logs.csv"),
    tf.keras.callbacks.TensorBoard(log_dir= LOG_DIR), 
]



hist = model.fit(train_ds, epochs = args.epochs, callbacks=callbacks, initial_epoch=0, )

back_bone.save(os.path.join(CKPT_DIR,"final_model"))


