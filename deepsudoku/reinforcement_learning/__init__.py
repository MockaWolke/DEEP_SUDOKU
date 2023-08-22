import gymnasium
import warnings

# This will silence the specific UserWarning with the message containing "A Box observation space"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="gymnasium.utils.passive_env_checker",
    message=".*A Box observation space.*",
)

from deepsudoku.reinforcement_learning.env import create_sudoku_env_v0, create_sudoku_env_v1

gymnasium.register(
    id="Sudoku-v0",
    entry_point=create_sudoku_env_v0,
    kwargs={
        "difficulty": "easy",
        "factor_in_density": False,
        "upper_bound_missing_digist": None,
        "render_mode": "human",
    },
    max_episode_steps=81,
)

gymnasium.register(
    id="Sudoku-v1",
    entry_point=create_sudoku_env_v1,
    kwargs={
        "size": 3,
        "render_mode": "none"
    },
    max_episode_steps=81,
)


print("Sudoku Environment avaible at gymnasium as 'Sudoku-v0'.")
