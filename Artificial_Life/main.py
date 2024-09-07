from env_grid import EnvironmentGrid
from decision import StentorRoeseli
from visualize import  visualize_grid


def get_input(prompt, type_=int, default=None):
    try:
        return type_(input(prompt))
    except ValueError:
        return default

def get_bead_positions():
    bead_positions = []
    print("Enter bead positions in the format (row, col, size). Type 'done' when finished:")
    while True:
        bead_input = input("Bead position: ")
        if bead_input.lower() == 'done':
            break
        try:
            row, col, distance = map(float, bead_input.split(','))
            bead_positions.append(((int(row), int(col)), distance))
        except ValueError:
            print("Invalid input. Please enter the values in the correct format.")
    return bead_positions

def simulate_stentor_environment():
    total_rows = get_input("Enter the total number of rows for the grid: ")
    total_cols = get_input("Enter the total number of columns for the grid: ")
    stentor_row = get_input("Enter the row for the Stentor: ")
    stentor_col = get_input("Enter the column for the Stentor: ")
    day_to_simulate = get_input("Enter the day of the year to simulate: ")
    bead_positions = get_bead_positions()

    environment_grid = EnvironmentGrid(rows=total_rows, cols=total_cols)
    
    for day in range(day_to_simulate):
        environment_grid.update_grid()

    stentor = StentorRoeseli(environment_grid=environment_grid, row=stentor_row, col=stentor_col)
    

    
    stentor_action = 'rest'
    
    visualize_grid(stentor, environment_grid, bead_positions, stentor_action)
    
    while stentor_action != 'detach':
        stentor_action = stentor.respond_to_bead_stimulus(0.0002, 1, 0.001)
        
        environment_grid.update_grid()
        
        visualize_grid(stentor, environment_grid, bead_positions, stentor_action)
        
        stentor.manage_energies()
        
    

simulate_stentor_environment()
