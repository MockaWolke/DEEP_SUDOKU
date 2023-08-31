import numpy as np
from deepsudoku.generate import Generator, Solver
from deepsudoku import exp_generate
from deepsudoku.utils import string_to_array, visualize_sudoku
import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class SudokuEnv_v0(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        difficulty,
        factor_in_density=False,
        upper_bound_missing_digist=None,
        render_mode="human",
    ):
        self.difficulty = difficulty
        self.factor_in_density = factor_in_density
        self.upper_bound_missing_digist = upper_bound_missing_digist
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)

        self.reward_range = (-1, 1)

        self._generate_field()

    def _generate_field(self):
        solver = Solver()
        generator = Generator(
            "9", self.difficulty, solver, self.upper_bound_missing_digist
        )

        quiz, solution = generator.generate_one(self.factor_in_density)

        self.field = string_to_array(quiz).astype(np.int32)
        self.solution = string_to_array(solution).astype(np.int32)

    def reset(self, seed=None, options=None):
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

        return (
            self.field,
            reward,
            terminated,
            False,
            {},
        )  # Returning observation, reward, terminated, truncated, and info dictionary

    def render(self):
        if self.render_mode == "human":
            visualize_sudoku(self.field)

        elif self.render_mode == "rgb_array":
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis("tight")
            plt.axis("off")
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170, 200:-170]

    def close(self):
        # Implement cleanup logic if needed
        pass


class SudokuEnv_x1(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        difficulty,
        factor_in_density=False,
        upper_bound_missing_digist=None,
        render_mode="human",
    ):
        self.difficulty = difficulty
        self.factor_in_density = factor_in_density
        self.upper_bound_missing_digist = upper_bound_missing_digist
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)

        self.reward_range = (-1, 100)

        self._generate_field()

    def _generate_field(self):
        solver = Solver()
        generator = Generator(
            "9", self.difficulty, solver, self.upper_bound_missing_digist
        )

        quiz, solution = generator.generate_one(self.factor_in_density)

        self.field = string_to_array(quiz).astype(np.int32)
        self.solution = string_to_array(solution).astype(np.int32)

    def reset(self, seed=None, options=None):
        self._generate_field()
        return self.field, {}  # Returning observation and info dictionary

    def step(self, action):
        y, x, number = action

        number += 1

        reward = 0
        terminated = False

        if self.field[y, x] == 0 and self.solution[y, x] == number:
            reward = 1

            self.field[y, x] = number

            if np.array_equal(self.field, self.solution):
                terminated = True
                reward = 100

        elif self.field[y, x] == 0:
            reward = 0.5
            terminated = True

        else:
            reward = -1
            terminated = True

        return (
            self.field,
            reward,
            terminated,
            False,
            {},
        )  # Returning observation, reward, terminated, truncated, and info dictionary

    def render(self):
        if self.render_mode == "human":
            visualize_sudoku(self.field)

        elif self.render_mode == "rgb_array":
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis("tight")
            plt.axis("off")
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170, 200:-170]

    def close(self):
        # Implement cleanup logic if needed
        pass


class SudokuEnv_x2(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        difficulty,
        factor_in_density=False,
        upper_bound_missing_digist=None,
        render_mode="human",
        easy_fraq=0,
        easy_start=15,
        easy_mode="range",
    ):
        self.difficulty = difficulty
        self.factor_in_density = factor_in_density
        self.upper_bound_missing_digist = upper_bound_missing_digist
        self.render_mode = render_mode
        self.easy_fraq = easy_fraq
        self.easy_start = easy_start
        self.easy_mode = easy_mode

        assert self.easy_mode in ["range", "precise"]

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)

        self.reward_range = (-1, 1)

        self._generate_field()

    def _generate_field(self):
        solver = Solver()

        upper_bound = self.upper_bound_missing_digist

        if self.easy_fraq != 0:
            if np.random.uniform(0, 1) < self.easy_fraq:
                upper_bound = (
                    np.random.randint(1, self.easy_start + 1)
                    if self.easy_mode == "range"
                    else self.easy_start
                )

        generator = Generator("9", self.difficulty, solver, upper_bound)

        quiz, solution = generator.generate_one(self.factor_in_density)

        self.field = string_to_array(quiz).astype(np.int32)
        self.solution = string_to_array(solution).astype(np.int32)

    def reset(self, seed=None, options=None):
        self._generate_field()
        return self.field, {}  # Returning observation and info dictionary

    def step(self, action):
        y, x, number = action

        number += 1

        reward = 0
        terminated = False

        if self.field[y, x] == 0 and self.solution[y, x] == number:
            reward = 1

            self.field[y, x] = number

            if np.array_equal(self.field, self.solution):
                terminated = True

        elif self.field[y, x] == 0:
            reward = 0.5
            terminated = True

        else:
            reward = -1
            terminated = True

        return (
            self.field,
            reward,
            terminated,
            False,
            {},
        )  # Returning observation, reward, terminated, truncated, and info dictionary

    def render(self):
        if self.render_mode == "human":
            visualize_sudoku(self.field)

        elif self.render_mode == "rgb_array":
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis("tight")
            plt.axis("off")
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170, 200:-170]

    def close(self):
        # Implement cleanup logic if needed
        pass


class SudokuEnv_nostop0(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        difficulty,
        factor_in_density=False,
        upper_bound_missing_digist=None,
        render_mode="human",
        easy_fraq=0,
        easy_start=15,
        easy_mode="range",
        use_random_starting_point=True,
        cut_off_limit=10,
    ):
        assert difficulty == "easy"

        self.difficulty = difficulty
        self.factor_in_density = factor_in_density
        self.upper_bound_missing_digist = upper_bound_missing_digist
        self.render_mode = render_mode
        self.easy_mode = easy_mode

        assert self.easy_mode in ["range", "precise"]

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)

        self.reward_range = (-1, 1)

        self.solver = Solver()
        self.generator = Generator(
            "9", self.difficulty, self.solver, None, use_random_starting_point
        )
        self.cut_off_limit = cut_off_limit

        self.fail_counter = 0

        self._generate_field()

    def _generate_field(self):
        upper_bound = self.upper_bound_missing_digist
        
        
        quiz, solution = self.generator.generate_one(
            self.factor_in_density, upper_bound
        )

        self.field = string_to_array(quiz).astype(np.int32)
        self.solution = string_to_array(solution).astype(np.int32)

        self.fail_counter = 0

    def reset(self, seed=None, options=None):
        self._generate_field()
        return self.field, {}  # Returning observation and info dictionary

    def step(self, action):
        y, x, number = action

        number += 1

        reward = 0
        terminated = False

        if self.field[y, x] == 0 and self.solution[y, x] == number:
            reward = 3

            self.field[y, x] = number
            self.fail_counter = 0

            if np.array_equal(self.field, self.solution):
                terminated = True


        else:
            reward = -0.1
            self.fail_counter += 1

            terminated = self.fail_counter >= self.cut_off_limit

        return (
            self.field,
            reward,
            terminated,
            False,
            {},
        )  # Returning observation, reward, terminated, truncated, and info dictionary

    def render(self):
        if self.render_mode == "human":
            visualize_sudoku(self.field)

        elif self.render_mode == "rgb_array":
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis("tight")
            plt.axis("off")
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170, 200:-170]

    def close(self):
        # Implement cleanup logic if needed
        pass


class SudokuEnv_x1n(SudokuEnv_x1):
    def __init__(
        self,
        difficulty,
        factor_in_density=False,
        upper_bound_missing_digist=None,
        render_mode="human",
    ):
        super().__init__(
            difficulty, factor_in_density, upper_bound_missing_digist, render_mode
        )

    def step(self, action):
        field, reward, terminated, trunated, info = super().step(action)

        reward = ((reward + 1) / 55.5) - 1

        return field, reward, terminated, trunated, info


class SudokuEnv_x0(gymnasium.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 1}

    def __init__(self, size=3, render_mode="none", missing_digits=3):
        self.render_mode = render_mode
        self.size = size
        self.square_size = np.square(self.size)
        self.missing_digits = missing_digits

        # Define action and observation spaces
        # Discrete Action space
        self.action_space = spaces.Discrete(self.square_size**3)
        self.observation_space = spaces.Box(
            low=0,
            high=self.square_size,
            shape=(self.square_size, self.square_size),
            dtype=np.int32,
        )
        self.reward_range = (-1, 1)

        self.generator = exp_generate.Generator(self.size)
        self._generate_field()

    def _generate_field(self):
        self.field, self.solution = self.generator.generate_one(
            missing_digits=self.missing_digits
        )

    def reset(self, seed=None, options=None):
        self._generate_field()
        return self.field, {}  # Returning observation and info dictionary

    def step(self, action):
        # Discrete Action space:
        # We use a base-square_size number system:
        # First (square_size) numbers x correspond to y=0, x=0, number = x
        # The (square_size) numbers x after correspond to y=0 , x=1, number = x-(square_size*1) etc...
        y = action // (self.square_size**2)
        x = (action % (self.square_size**2)) // (self.square_size)
        number = action % (self.square_size)

        number += 1

        reward = 0
        terminated = False

        if self.field[y, x] == 0 and self.solution[y, x] == number:
            reward = 1

            self.field[y, x] = number

            if np.array_equal(self.field, self.solution):
                terminated = True
                reward = 1

        else:
            reward = -1
            terminated = True

        return (
            self.field,
            reward,
            terminated,
            False,
            {},
        )  # Returning observation, reward, terminated, truncated, and info dictionary

    def render(self):
        if self.render_mode == "human":
            visualize_sudoku(self.field)

        elif self.render_mode == "rgb_array":
            fig = visualize_sudoku(self.field, return_fig=True, figsize=(6, 6), dpi=300)
            plt.axis("tight")
            plt.axis("off")
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data[200:-170, 200:-170]

    def close(self):
        # Implement cleanup logic if needed
        pass


def create_sudoku_env_x1n(
    difficulty,
    factor_in_density=False,
    upper_bound_missing_digist=None,
    render_mode="human",
):
    return SudokuEnv_x1n(
        difficulty,
        factor_in_density,
        upper_bound_missing_digist=upper_bound_missing_digist,
        render_mode=render_mode,
    )


def create_sudoku_env_v0(
    difficulty,
    factor_in_density=False,
    upper_bound_missing_digist=None,
    render_mode="human",
):
    return SudokuEnv_v0(
        difficulty,
        factor_in_density,
        upper_bound_missing_digist=upper_bound_missing_digist,
        render_mode=render_mode,
    )


def create_sudoku_env_v1(render_mode="none", size=3, missing_digits=3):
    return SudokuEnv_x0(
        size=size, render_mode=render_mode, missing_digits=missing_digits
    )


def create_sudoku_env_x1(
    difficulty,
    factor_in_density=False,
    upper_bound_missing_digist=None,
    render_mode="human",
):
    return SudokuEnv_x1(
        difficulty,
        factor_in_density,
        upper_bound_missing_digist=upper_bound_missing_digist,
        render_mode=render_mode,
    )


def create_sudoku_env_x2(
    difficulty,
    factor_in_density=False,
    upper_bound_missing_digist=None,
    render_mode="human",
    easy_fraq=0,
    easy_start=15,
    easy_mode="range",
):
    return SudokuEnv_x2(
        difficulty,
        factor_in_density,
        upper_bound_missing_digist=upper_bound_missing_digist,
        render_mode=render_mode,
        easy_fraq=easy_fraq,
        easy_start=easy_start,
        easy_mode=easy_mode,
    )


def create_sudoku_nostop0(
    difficulty,
    factor_in_density=False,
    upper_bound_missing_digist=None,
    render_mode="human",
    easy_fraq=0,
    easy_start=15,
    easy_mode="range",
    use_random_starting_point=True,
    cut_off_limit=10,
):
    return SudokuEnv_nostop0(
        difficulty,
        factor_in_density=factor_in_density,
        upper_bound_missing_digist=upper_bound_missing_digist,
        render_mode=render_mode,
        easy_fraq=easy_fraq,
        easy_start=easy_start,
        easy_mode=easy_mode,
        use_random_starting_point=use_random_starting_point,
        cut_off_limit=cut_off_limit,
    )
