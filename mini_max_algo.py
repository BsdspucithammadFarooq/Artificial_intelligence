class Minimax:
    def __init__(self, game_state):
        self.game_state = game_state

    def is_terminal(self, state):
        """Check if the game has ended """
        return self.check_winner(state) or not any(' ' in row for row in state)

    def check_winner(self, state):
        """THis will ccheck all the possible winning combinations roes cols and 2 diagonals"""
        for row in state:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return row[0]
        for col in range(3):
            if state[0][col] == state[1][col] == state[2][col] and state[0][col] != ' ':
                return state[0][col]
        if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
            return state[0][0]
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
            return state[0][2]
        return None  # No winner yet

    def utility(self, state):
        # Utility values: +1 for 'X' win, -1 for 'O' win, 0 for draw
        winner = self.check_winner(state)
        if winner == 'X':
            return 1  # Maximizer wins
        elif winner == 'O':
            return -1  # Minimizer wins
        else:
            return 0  # Draw

    def minimax(self, state, depth, maximizing_player):
        """base case if the gamee is over return the utility value"""
        if self.is_terminal(state):
            return self.utility(state)

        if maximizing_player:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':  # Empty cell
                        state[i][j] = 'X'  # Make the move
                        eval = self.minimax(state, depth + 1, False)  # Recur for minimizer
                        state[i][j] = ' '  # Undo the move
                        max_eval = max(max_eval, eval)
            return max_eval
        else:  # Minimizer's turn (Player 'O')
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':  # Empty,move
                        state[i][j] = 'O'
                        eval = self.minimax(state, depth + 1, True)  # Recur for maximizer
                        state[i][j] = ' '  # Undo the move
                        min_eval = min(min_eval, eval)
            return min_eval

    def best_move(self, state):
        """determivne the best move for the maximizer 0 or x"""
        best_val = float('-inf')
        best_move = None

        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':  # Empty, move
                    state[i][j] = 'X'
                    move_val = self.minimax(state, 0, False)  # Minimize for 'O'
                    state[i][j] = ' '  # Undo the move

                    if move_val > best_val:  # Find the move with the best value
                        best_val = move_val
                        best_move = (i, j)

        return best_move


board = [
    ['X', 'O', 'X'],
    ['O', 'X', ' '],
    [' ', ' ', 'O']
]
game = Minimax(board)
# Find the best move for Player 'X'
best = game.best_move(board)
print(f"Best move for 'X': {best}")
