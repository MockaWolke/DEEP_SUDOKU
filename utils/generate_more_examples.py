import os
from deepsudoku import REPO_PATH, DIFFICULTIES, generate
from itertools import product
import tqdm

os.chdir(REPO_PATH)


# define the number of CPUs to use

NUM_CPUS = 6
WANTED = {
    "easy" : 100000,
    "medium" : 100000,
    "extreme" : 100000,
    "hard" : 100000,
    "insane" : 500000,
}


SETTINGS = list(product(DIFFICULTIES, [True, False]))


def generate_and_save_data(setting, number):  
    
    difficulty, density = setting
    
    density_name = "considering_density" if density else "without_density"
    
    dir_path = os.path.join(REPO_PATH, "data", f"{difficulty}_{density_name}")
    os.makedirs(dir_path, exist_ok= True)
    
    result_path = os.path.join(dir_path, f"chunk_{number}.txt")
    
    if os.path.exists(result_path):
        return
    
    examples = generate.generate_many_games(STEP_SIZE, NUM_CPUS, difficulty, density)

    lines = [" ".join(line)+"\n" for line in examples]


    with open(result_path, 'w') as f:
        f.writelines(lines)
        
if __name__ == "__main__":
    
    assert all([i in WANTED for i in DIFFICULTIES])
    
    for setting in SETTINGS:
        
        
        STEP_SIZE = 100000
        EXAMPLES_PER_DIFFICULTY = WANTED[setting[0]]

        N_STEPS = EXAMPLES_PER_DIFFICULTY // STEP_SIZE

        
        print("Now Doing Setting:", setting)
        
        for number in tqdm.tqdm(range(N_STEPS)):
            
            generate_and_save_data(setting, number)
        