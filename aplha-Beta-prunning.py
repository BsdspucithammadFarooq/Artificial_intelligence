class AlphaBetaPruning:
    def __init__(self, depth, game_state, player):
        # Initialize the depth, current game state, and player (maximizer or minimizer)
        self.depth = depth
        self.game_state = game_state
        self.player = player  # 'X' for maximizer, 'O' for minimizer

    def is_terminal(self, state):
        # Check if the game has reached a terminal state (win, lose, draw)
        # Returns True if the game is over, False otherwise
        if self.check_winner(state, 'X') or self.check_winner(state, 'O') or self.is_draw(state):
            return True
        return False

    def utility(self, state):
        """Utitlties for min_max alpha beta prunning"""
        if self.check_winner(state, 'X'):
            return 1  # Max X wins
        elif self.check_winner(state, 'O'):
            return -1  #min O wins
        else:
            return 0 #d
    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        """This is the main algo for alpha-beta prunning which is the advance version of mini-mox algo -_-"""
        if depth == 0 or self.is_terminal(state):
            return self.utility(state)

        if maximizing_player:
            max_eval = -float('inf')
            for move in self.get_available_moves(state):
                new_state = self.make_move(state, move, 'X')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_available_moves(state):
                new_state = self.make_move(state, move, 'O')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def best_move(self, state):
        """This function will choose the best move"""
        best_val = -float('inf')
        best_move = None
        for move in self.get_available_moves(state):
            new_state = self.make_move(state, move, 'X')
            move_val = self.alphabeta(new_state, self.depth - 1, -float('inf'), float('inf'), False)
            if move_val > best_val:
                best_val = move_val
                best_move = move
        return best_move
    def check_winner(self, state, player):
        """possible no of combinations of the winner, rows and cols and diagonals :)"""
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        for condition in win_conditions:
            if state[condition[0]] == state[condition[1]] == state[condition[2]] == player:
                return True
        return False
    def is_draw(self, state):
        """Gmae is draw if there are no empty spaces left"""
        return all(cell != ' ' for cell in state)
    def get_available_moves(self, state):
        """check for the no of available moves """
        return [i for i in range(len(state)) if state[i] == ' ']
    def make_move(self, state, move, player):
        """This function will just make the move on the state and return new state"""
        new_state = state[:]
        new_state[move] = player
        return new_state
def main():
    """THis is the main function which is calling the ALpha-Beta Mini_max algo"""
    initial_state = ['O', ' ', 'O', ' ', ' ', ' ', 'X', ' ', 'X']
    ai = AlphaBetaPruning(depth=3, game_state=initial_state, player='O')
    best_move = ai.best_move(initial_state)
    print("Best move for AI:", best_move)
main()
