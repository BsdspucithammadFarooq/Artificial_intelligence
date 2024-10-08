def manhattan_distance(state, goal):
    "THis function will calculate the total distance of misplaced tiles from curent state to gaol state"
    distance = 0
    for i in range(9):
        if state[i] != 0:  # Ignore empty space
            current_row = i // 3
            current_col = i % 3

            goal_index = goal.index(state[i])
            goal_row = goal_index // 3
            goal_col = goal_index % 3

            row_diff = current_row - goal_row
            if row_diff < 0:
                row_diff = -row_diff# to be +ve
            col_diff = current_col - goal_col
            if col_diff < 0:
                col_diff = -col_diff #te be +ve

            distance += row_diff + col_diff
    return distance



def misplaced_tiles(state, goal):
    """THis function will give the total no of misplaced titles :)"""
    count = 0
    for i in range(9):
        if state[i] != 0 and state[i] != goal[i]:#ignore the empty tile(0)
            count += 1
    return count
initial_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

mis = misplaced_tiles(initial_state, goal_state)
print("Misplaced Tiles:", mis)

manhat = manhattan_distance(initial_state, goal_state)
print("Manhattan Distance:", manhat)