#pip install nashpy

import nashpy as nash

# Define the payoff matrices for row player and column player
row_matrix = [[2, 0], [0, 1]]
col_matrix = [[1, 0], [0, 2]]

# Create a Normal Form Game
game = nash.Game(row_matrix, col_matrix)

# Compute Nash equilibria
equilibria = game.support_enumeration()
print("Nash equilibria:", list(equilibria))

