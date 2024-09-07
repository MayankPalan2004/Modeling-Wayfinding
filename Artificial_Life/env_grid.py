from env import Environment

class EnvironmentGrid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[Environment() for _ in range(cols)] for _ in range(rows)]

    def update_grid(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].update()

    def get_grid_state(self):
        grid_state = []
        for row in range(self.rows):
            row_state = []
            for col in range(self.cols):
                row_state.append(self.grid[row][col].get_state())
            grid_state.append(row_state)
        return grid_state

    def get_adjacent_grids(self, row, col):
        adjacent_grids = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

        for dr, dc in directions:
            adj_row, adj_col = row + dr, col + dc
            if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                adjacent_grids.append((adj_row, adj_col, self.grid[adj_row][adj_col]))

        return adjacent_grids



    
    
