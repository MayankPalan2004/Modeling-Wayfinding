import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors

def visualize_grid(stentor, environment_grid, bead_positions, stentor_action):
    fig, ax = plt.subplots()

    # Create a colormap for the environment scores
    cmap = colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    # Plot the grid and shade cells based on env_score
    for row in range(environment_grid.rows):
        for col in range(environment_grid.cols):
            env_state = environment_grid.grid[row][col].get_state()
            env_score = stentor.assess_environment(env_state)
            rect_color = cmap(env_score)  # Convert env_score to a color
            rect = patches.Rectangle((col, row), 1, 1, edgecolor='black', facecolor=rect_color)
            ax.add_patch(rect)

    # Plot the Stentor
    stentor_row, stentor_col = stentor.row, stentor.col
    ax.add_patch(patches.Circle((stentor_col + 0.5, stentor_row + 0.5), 0.3, color='blue', label="Stentor"))

    # Plot the beads with varying sizes
    for bead_pos, bead_size in bead_positions:
        bead_row, bead_col = bead_pos
        ax.add_patch(patches.Circle((bead_col + 0.5, bead_row + 0.5), bead_size * 0.1, color='red', label="Bead"))

    # Highlight the action taken by the Stentor
    if stentor_action == "bend":
        ax.add_patch(patches.Circle((stentor_col + 0.5, stentor_row + 0.5), 0.5, edgecolor='blue', facecolor='none'))
    elif stentor_action == "alternate_cilia":
        ax.add_patch(patches.Rectangle((stentor_col, stentor_row), 1, 1, edgecolor='green', facecolor='none', linewidth=2))
    elif stentor_action == "contract":
        ax.add_patch(patches.Circle((stentor_col + 0.5, stentor_row + 0.5), 0.15, color='blue'))
    elif stentor_action == "detach":
        ax.add_patch(patches.Circle((stentor_col + 0.5, stentor_row + 0.5), 0.4, edgecolor='purple', facecolor='none', linestyle='--'))

    # Set limits and display
    ax.set_xlim(0, environment_grid.cols)
    ax.set_ylim(0, environment_grid.rows)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
    plt.show()


