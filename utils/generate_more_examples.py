import os
from deepsudoku import REPO_PATH, DIFFICULTIES

os.chdir(REPO_PATH)


# define the number of CPUs to use
from multiprocessing import Pool
from deepsudoku.generate import generate
import numpy as np
import tqdm
from hashlib import sha256
import json

STEP_SIZE = 5000
EXAMPLES_PER_DIFFICULTY = int(1e6/ 2)
NUM_CPUS = 8
N_STEPS = EXAMPLES_PER_DIFFICULTY // STEP_SIZE
N_STEPS

def generate_and_save_data(args):
    
    difficulty, number = args
    
    dir_path = os.path.join(REPO_PATH, "raw_data", difficulty)
    os.makedirs(dir_path, exist_ok= True)
    
    input_path = os.path.join(dir_path, f"inputs_{number:05d}.npy")
    
    if os.path.exists(input_path):
        return
    
    examples = generate(difficulty = difficulty, verbose = 0, count = STEP_SIZE)

    inputs = np.stack([a[0] for a in examples])
    labels = np.stack([a[1] for a in examples])

    np.save(input_path, inputs)
    
    label_path = os.path.join(dir_path, f"labels_{number:05d}.npy")
    
    np.save(label_path, labels)



with Pool(NUM_CPUS) as p:
    for difficulty in DIFFICULTIES[::-1]:
        args = [(difficulty, i) for i in range(N_STEPS // 2, N_STEPS)]
        list(tqdm.tqdm(p.imap(generate_and_save_data, args), total= N_STEPS // 2))




