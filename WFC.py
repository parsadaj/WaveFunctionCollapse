import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from copy import deepcopy

# Step 1: Define a custom exception class
class NoWFCSolution(Exception):
    def __init__(self, message="No WFC solutions Found!"):
        super().__init__(message)
        
        
class WaveFunctionCollapse:
    def __init__(self, input_image, pattern_size, grid_size=None, random_seed=None):
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = 0
        self.input_image = np.array(input_image)
        self.input_size = self.input_image.shape
        self.pattern_size = pattern_size
        # self.patterns = []
        self.pattern_frequencies = defaultdict(int)
        self.adjacency_rules = defaultdict(lambda: defaultdict(set))
        
        if grid_size is None:
            self.grid_size = self.input_size
        else:
            self.grid_size = grid_size
        self.output_grid = None
        self.possible_patterns = None
        self.observations = []
        

    def generate_pattern_mappings(self):
        # Generate a dictionary mapping each pattern to a unique number
        pattern_to_number = {pattern: idx for idx, pattern in enumerate(self.patterns)}
        
        # Generate the reverse mapping from number to pattern
        number_to_pattern = {idx: pattern for pattern, idx in pattern_to_number.items()}
        
        return pattern_to_number, number_to_pattern
    
    def extract_patterns(self):
        """Extract patterns and compute their frequencies and adjacency rules."""
        height, width = self.input_image.shape
        for y in range(height):
            for x in range(width):
                pattern = self.get_pattern(x, y)
                # self.patterns.append(pattern)
                self.pattern_frequencies[tuple(pattern.flatten())] += 1

        self.patterns = self.pattern_frequencies.keys()  # Remove duplicates
        self.pattern_to_number, self.number_to_pattern = self.generate_pattern_mappings()

        # Set up adjacency rules
        for y in range(height):
            for x in range(width):
                pattern = self.get_pattern(x, y)
                above_pattern = self.get_pattern(x, (y - 1) % height)
                below_pattern = self.get_pattern(x, (y + 1) % height)
                left_pattern = self.get_pattern((x - 1) % width, y)
                right_pattern = self.get_pattern((x + 1) % width, y)

                self.adjacency_rules[tuple(pattern.flatten())]['above'].add(tuple(above_pattern.flatten()))
                self.adjacency_rules[tuple(pattern.flatten())]['below'].add(tuple(below_pattern.flatten()))
                self.adjacency_rules[tuple(pattern.flatten())]['left'].add(tuple(left_pattern.flatten()))
                self.adjacency_rules[tuple(pattern.flatten())]['right'].add(tuple(right_pattern.flatten()))

    def get_pattern(self, x, y):
        """Get a pattern from the input image, considering wrapping."""
        pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=self.input_image.dtype)
        for dy in range(self.pattern_size):
            # if y + dy >= self.input_size[0]:
            #     break
            for dx in range(self.pattern_size):
                # if x + dx >= self.input_size[1]:
                #     break
                pattern[dy, dx] = self.input_image[(y + dy) % self.input_size[0], (x + dx) % self.input_size[1]]
                # pattern[dy, dx] = self.input_image[(y + dy), (x + dx)]
        return pattern

    def initialize_output_grid(self):
        """Initialize the output grid with all possible patterns."""
        self.output_grid = np.full(self.grid_size, None)
        self.possible_patterns = [[set(self.pattern_frequencies.keys()) for _ in range(self.grid_size[1])]
                                  for _ in range(self.grid_size[0])]

    def observe(self):
        """Select the most constrained cell and collapse its wave function."""
        min_entropy = float('inf')
        chosen_cell = None
        chosen_cells = []
        
        self.observations.append(deepcopy(self.possible_patterns))

        # Find the cell with the least possibilities (highest constraint)
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if self.output_grid[y, x] is None:
                    num_possibilities = len(self.possible_patterns[y][x])
                    if num_possibilities < min_entropy:
                        chosen_cells = []
                        min_entropy = num_possibilities
                        chosen_cell = (y, x)
                        chosen_cells.append(chosen_cell)
                    elif num_possibilities == min_entropy:
                        chosen_cell = (y, x)
                        chosen_cells.append(chosen_cell)
                        
        if chosen_cell is None or len(chosen_cells) == 0:
            return False  # All cells are filled

        if min_entropy == 0:
            raise NoWFCSolution
        
        chosen_cell = random.choice(chosen_cells)
        
        y, x = chosen_cell
        # Randomly select a pattern weighted by its frequency
        pattern = random.choices(
            list(self.possible_patterns[y][x]),
            weights=[self.pattern_frequencies[pat] for pat in self.possible_patterns[y][x]],
            k=1
        )[0]
        self.output_grid[y, x] = pattern
        self.possible_patterns[y][x] = {pattern}

        self.observations.append(deepcopy(self.possible_patterns))

        return True

    def propagate(self):
        """Propagate constraints from the observed cell throughout the grid."""
        changes = True
        while changes:
            changes = False
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    if len(self.possible_patterns[y][x]) == 1:
                        pattern = next(iter(self.possible_patterns[y][x]))
                        for direction, dy, dx in [('above', -1, 0), ('below', 1, 0), ('left', 0, -1), ('right', 0, 1)]:
                            # if y + dy >= self.grid_size[0] or x + dx >= self.grid_size[1]:
                            #     continue
                            
                            ny, nx = (y + dy) % self.grid_size[0], (x + dx) % self.grid_size[1]
                            # ny, nx = (y + dy), (x + dx)
                            valid_patterns = set()
                            for neighbor_pattern in self.possible_patterns[ny][nx]:
                                if neighbor_pattern in self.adjacency_rules[pattern][direction]:
                                    valid_patterns.add(neighbor_pattern)
                            if valid_patterns != self.possible_patterns[ny][nx]:
                                self.possible_patterns[ny][nx] = valid_patterns
                                changes = True

    def run(self):
        try:
            """Run the WFC algorithm until the grid is fully populated."""
            self.extract_patterns()
            self.initialize_output_grid()

            while True:
                if not self.observe():
                    break
                self.propagate()

            # Construct the final output image
            final_image = np.zeros(self.grid_size, dtype=self.input_image.dtype)
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    pattern = self.output_grid[y, x]
                    final_image[y, x] = np.array(pattern).reshape(self.pattern_size, self.pattern_size)[0, 0]

            return final_image
        except NoWFCSolution:
            print("No Solution Found, Retrying...")
            return self.run()


class WaveFunctionCollapseVisualizer:
    def __init__(self, grid_size, observations, pattern_to_number, pattern_size, adjacency_rules, cmap=None):
        self.pattern_size = pattern_size
        self.grid_size = grid_size
        self.observations = observations
        self.pattern_to_number = pattern_to_number
        if cmap is not None:
            self.custom_colormap = cmap
        else:
            self.custom_colormap = 'terrain'
        self.adjacency_rules = adjacency_rules  # Adjacency rules: a dictionary
        
    def plot_observations(self):
        n = len(self.observations)
        fig, axes = plt.subplots(n,1, figsize=(20, n* 4))
        if n == 1:
            axes = [axes]  # Handle the case where n=1

        for idx, observation in enumerate(self.observations):
            ax = axes[idx]
            grid = np.zeros(self.grid_size, dtype=object)

            # Fill grid with pattern numbers
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    patterns = observation[i][j]
                    grid[i, j] = ','.join(str(self.pattern_to_number[p]) for p in patterns)

            # Plot the grid
            ax.set_xticks(np.arange(0, self.grid_size[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(0, self.grid_size[0] + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
            ax.tick_params(which="minor", size=0)
            ax.axis('off')  # Turn off the axis

            # Add pattern numbers in each cell
            for (i, j), label in np.ndenumerate(grid):
                ax.text(j, i, label, ha='center', va='center', fontsize=8)

            # ax.set_title(f"Observation {idx + 1}")

        plt.tight_layout()

    def visualize_patterns(self):
        unique_patterns = list(self.pattern_to_number.keys())
        n = len(unique_patterns)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]  # Handle the case where n=1

        for idx, pattern in enumerate(unique_patterns):
            ax = axes[idx]
            pattern_array = np.array(pattern)  # Assuming patterns can be represented as 2D arrays

            ax.imshow(pattern_array.reshape(self.pattern_size, self.pattern_size), cmap=self.custom_colormap, interpolation="nearest", vmin=0, vmax=2)
            ax.axis('off')  # Turn off the axis
            # ax.set_title(f"Pattern {self.pattern_to_number[pattern]}")
            ax.text(0, 0, self.pattern_to_number[pattern], ha='center', va='center', fontsize=32, color='white')
            
        plt.tight_layout()
            
    def visualize_adjacency(self, pattern):
        pattern_number = self.pattern_to_number[pattern]

        # Initialize a dynamic grid to accommodate multiple adjacent patterns
        # For now, start with 3x3 and expand as necessary
        grid_size = 3
        pattern_shape = (self.pattern_size, self.pattern_size)
        
        # Start with empty lists to hold patterns for each side
        above_patterns = []
        below_patterns = []
        left_patterns = []
        right_patterns = []
        
        # Get adjacency rules for this pattern (if they exist)
        if pattern in self.adjacency_rules:
            rules = self.adjacency_rules[pattern]
            # Collect all patterns for each direction
            if 'above' in rules:
                above_patterns.extend(rules['above'])
            if 'below' in rules:
                below_patterns.extend(rules['below'])
            if 'left' in rules:
                left_patterns.extend(rules['left'])
            if 'right' in rules:
                right_patterns.extend(rules['right'])

        # Determine the grid dimensions needed to fit all patterns

        grid_rows = 1 + len(above_patterns) + len(below_patterns)  
        grid_cols = 1 + len(right_patterns) + len(left_patterns) 
        
        # Initialize an empty grid of pattern arrays
        grid = np.zeros((grid_rows, grid_cols, self.pattern_size*self.pattern_size), dtype=int)

        # Place the center pattern
        center_row, center_col = len(above_patterns), len(left_patterns)
        grid[center_row, center_col] = pattern

        # Helper function to place patterns in the grid
        def place_patterns(pattern_list, start_row, start_col, row_step, col_step):
            for idx, pat in enumerate(pattern_list):
                grid[start_row + idx * row_step, start_col + idx * col_step] = pat

        # Place above patterns
        place_patterns(above_patterns, 0, center_col, 1, 0)
        # Place below patterns
        place_patterns(below_patterns, center_row + 1, center_col, 1, 0)
        # Place left patterns
        place_patterns(left_patterns, center_row, 0, 0, 1)
        # Place right patterns
        place_patterns(right_patterns, center_row, center_col + 1, 0, 1)
        
        fig, ax = plt.subplots(grid_rows, grid_cols, figsize=(6, 6))

        # Plot all patterns in the grid
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == center_row or j == center_col:
                    ax[i,j].imshow(np.array(grid[i, j]).reshape(pattern_shape), cmap=self.custom_colormap, vmin=0, vmax=2,
                            interpolation="nearest")#, extent=(j, j + 1, i, i + 1))

                    # Add grid lines
                    ax[i,j].set_xticks(np.arange(-0.5, grid_cols), minor=True)
                    ax[i,j].set_yticks(np.arange(-0.5, grid_rows), minor=True)
                    ax[i,j].grid(which="minor", color="black", linestyle='-', linewidth=1.5)
                    ax[i,j].tick_params(which="minor", size=0)
                    ax[i,j].axis('off')  # Turn off the axis
                    ax[i,j].text(0, 0, self.pattern_to_number[tuple(grid[i, j])], ha='center', va='center', fontsize=10, color='white')
                else:
                    ax[i,j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                    ax[i,j].axis('off')  # Turn off the axis


        # Add the title
        plt.tight_layout()


if __name__ == "__main__":
    # Example Usage
    input_image = [
        [0, 1, 2, 0],
        [2, 0, 1, 1],
        [1, 2, 0, 2],
        [0, 1, 2, 0]
    ]
    wfc = WaveFunctionCollapse(input_image, pattern_size=2, grid_size=(12,12))
    output_image = wfc.run()
    print(output_image)
    
    
    # Example usage:
    # Assuming you have a grid size, a list of observations, and a mapping from patterns to numbers
    observations = wfc.observations

    pattern_to_number = wfc.pattern_to_number

    visualizer = WaveFunctionCollapseVisualizer(grid_size=(12,12), observations=observations, pattern_to_number=pattern_to_number, pattern_size=2, adjacency_rules=wfc.adjacency_rules)
    #visualizer.plot_observations()
    # visualizer.visualize_patterns()
    for pat in wfc.patterns:
        visualizer.visualize_adjacency(pat)


