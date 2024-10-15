class PuzzleNode:
    """This is just an initialized for the puzzle problem"""
    def __init__(self, state, parent, move, g_cost, h_cost):
        self.state = state
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
class AStarSolver:
    """This is a informed search algorithm using hurestic value to find to goal :)"""
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.open_list = []  #yet explore
        self.closed_list = []#DOne explored
    def solve(self):
        """THis will sovle the 8 puzzle problem"""
        start_node = PuzzleNode(self.start_state, None, None, 0, self.heuristic(self.start_state))
        self.open_list.append(start_node)
        while len(self.open_list) > 0:
            current_node = self.get_lowest_f_cost_node(self.open_list)
            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            self.open_list.remove(current_node)
            self.closed_list.append(current_node)
            #Generating childers
            children = self.generate_children(current_node)
            for child in children:
                if not self.node_in_list(self.closed_list, child):
                    if not self.node_in_list(self.open_list, child):
                        child.h_cost = self.heuristic(child.state)
                        child.f_cost = child.g_cost + child.h_cost
                        self.open_list.append(child)
        return None
    def generate_children(self, node):
        """THis fucntion will gegerate the possible siblings of a node"""
        children = []
        empty_index = node.state.index(0)
        row = empty_index // 3
        col = empty_index % 3
        #possible moves of the empty tile!
        moves = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:#THis check will check if  the move is out of the grid or not?
                new_index = new_row * 3 + new_col
                new_state = node.state[:]
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]  # Swap
                children.append(PuzzleNode(new_state, node, move[2], node.g_cost + 1, 0))
        return children
    def heuristic(self, state):
        """ThIS FUNCTION WILL CALULATE THE HURESTIC VALUE(mANHATTAN DISTANCE) and return the total distance from the goal state from the destination state"""
        distance = 0
        for i in range(1, len(state)):  # Tiles are numbered 1 to 8
            current_index = state.index(i)
            goal_index = self.goal_state.index(i)
            current_row = current_index // 3
            current_col = current_index % 3
            goal_row = goal_index // 3
            goal_col = goal_index % 3
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)#This will always calculate the +ve distance
        return distance
    def trace_solution(self, node):
        node3=node
        moves = []
        while node.parent is not None:

            moves.append(node.move)
            node = node.parent
        moves=moves[::-1]
        print(moves)
        self.print1(moves=moves,node=node3)
    def print1(self,moves,node):
        # print("yea")
        # print(node.parent)
        while node.parent is not None:
            # print(node.move)
            moves.append(node.move)
            node = node.parent
            if node.move == "left":
                # print("yes")
                i = self.start_state.index(0)
                print(i)
                # self.start_state[i], self.start_state[i + 1] = self.start_state[i + 1], self.start_state[i]
                # print(self.start_state)
    def get_lowest_f_cost_node(self, nodes):
        lowest_f_cost_node = nodes[0]
        for node in nodes:
            if node.f_cost < lowest_f_cost_node.f_cost:
                lowest_f_cost_node = node
        return lowest_f_cost_node
    def node_in_list(self, node_list, node):
        for n in node_list:
            if n.state == node.state:
                return True
        return False
    def is_solvable(self, state):
        """THis function will check if the problem is solvable or not if inversion=even,Solable"""
        inversions = 0
        state_list = [tile for tile in state if tile != 0]  # Ignore the empty tile
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] > state_list[j]:#If first is greater than the second there is an inversion
                    inversions += 1
        return inversions % 2 == 0#THe inversion needs to be eveen to be solable :)
def main():
    """This is the main which is calling A* ALGO"""
    start_state = [3, 1, 7, 8, 5, 0, 2, 4, 6]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    solver = AStarSolver(start_state, goal_state)
    if solver.is_solvable(start_state):
        solution = solver.solve()
        if solution:
            pass
            # print("Solution found! Moves:", solution)
            # solver.print1(solution)
            # print(solution)
        # else:
            # print("No solution exists.")
    # else:
    #     print("This puzzle is not solvable.")
main()