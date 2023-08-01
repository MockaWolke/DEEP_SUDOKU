import numpy as np 
from deepsudoku.generate import Generator, Solver
from deepsudoku.utils import string_to_array, visualize_sudoku
import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class SudokuEnv(gymnasium.Env):
    
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps' : 1}
    
    def __init__(self, difficulty, factor_in_density=False, upper_bound_missing_digist = None, render_mode = "human"):
        
        self.difficulty = difficulty
        self.factor_in_density = factor_in_density
        self.upper_bound_missing_digist = upper_bound_missing_digist
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)
        
        self.reward_range = (-1, 1)
        
        self._generate_field()
        
    def _generate_field(self):
        solver = Solver()
        generator = Generator('9', self.difficulty, solver, self.upper_bound_missing_digist)
      
        quiz, solution = generator.generate_one(self.factor_in_density)
        
        self.field = string_to_array(quiz).astype(np.int32)
        self.solution = string_to_array(solution).astype(np.int32)
    
    def reset(self, seed = None, options = None):
        self._generate_field()
        return self.field, {}  # Returning observation and info dictionary
    
    def step(self, action):
        y, x, number = action
        
        number += 1
        
        reward = 0
        terminated = False
        
        if self.field[y, x] == 0 and self.solution[y, x] == number:
            self.field[y, x] = number
            
            if np.array_equal(self.field, self.solution):
                terminated = True    
                reward = 1
        else:
            reward = -1
            terminated = True
        
        return self.field, reward, terminated, False, {}  # Returning observation, reward, terminated, truncated, and info dictionary
    
    def render(self):
        
        if self.render_mode == 'human':
            visualize_sudoku(self.field)
        
        elif self.render_mode == 'rgb_array':
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis('tight')
            plt.axis('off')
            fig.canvas.draw()
            
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170,200:-170]
 
    def close(self):
        # Implement cleanup logic if needed
        pass
    
    
def create_sudoku_env(difficulty, factor_in_density=False, upper_bound_missing_digist = None, render_mode = 'human'):
    return SudokuEnv(difficulty, factor_in_density,upper_bound_missing_digist = upper_bound_missing_digist, render_mode = render_mode)
