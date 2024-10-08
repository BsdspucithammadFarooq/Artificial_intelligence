import argparse
from collections import deque
class PuzzleState:
    def __init__(self, state, parent, move, depth, cost, key):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.key = key
        if self.state:
            self.map = ''.join(str(e) for e in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map

    def __str__(self):
        return str(self.map)

GoalState = [0, 1, 2, 3, 4, 5, 6, 7, 8]
GoalNode = None  # at finding solution
NodesExpanded = 0  # total nodes visited
MaxSearchDeep = 0  # max deep
MaxFrontier = 0  # max frontier

def bfs(startState):
    global MaxFrontier, GoalNode, MaxSearchDeep

    boardVisited = set()
    Queue = deque([PuzzleState(startState, None, None, 0, 0, 0)])

    while Queue:
        node = Queue.popleft()
        boardVisited.add(node.map)
        if node.state == GoalState:
            GoalNode = node
            return Queue
        posiblePaths = subNodes(node)
        for path in posiblePaths:
            if path.map not in boardVisited:
                Queue.append(path)
                boardVisited.add(path.map)
                if path.depth > MaxSearchDeep:
                    MaxSearchDeep = MaxSearchDeep + 1
        if len(Queue) > MaxFrontier:
            QueueSize = len(Queue)
            MaxFrontier = QueueSize
def dfs(startState):
    global MaxFrontier, GoalNode, MaxSearchDeep

    boardVisited = set()
    stack = list([PuzzleState(startState, None, None, 0, 0, 0)])
    while stack:
        node = stack.pop()
        boardVisited.add(node.map)
        if node.state == GoalState:
            GoalNode = node
            return stack
        # inverse the order of next paths for execution porpuses
        posiblePaths = reversed(subNodes(node))
        for path in posiblePaths:
            if path.map not in boardVisited:
                stack.append(path)
                boardVisited.add(path.map)
                if path.depth > MaxSearchDeep:
                    MaxSearchDeep = 1 + MaxSearchDeep
        if len(stack) > MaxFrontier:
            MaxFrontier = len(stack)


# Heuristic: distance to root numbers
values_0 = [0, 1, 2, 1, 2, 3, 2, 3, 4]
values_1 = [1, 0, 1, 2, 1, 2, 3, 2, 3]
values_2 = [2, 1, 0, 3, 2, 1, 4, 3, 2]
values_3 = [1, 2, 3, 0, 1, 2, 1, 2, 3]
values_4 = [2, 1, 2, 1, 0, 1, 2, 1, 2]
values_5 = [3, 2, 1, 2, 1, 0, 3, 2, 1]
values_6 = [2, 3, 4, 1, 2, 3, 0, 1, 2]
values_7 = [3, 2, 3, 2, 1, 2, 1, 0, 1]
values_8 = [4, 3, 2, 3, 2, 1, 2, 1, 0]


def Heuristic(node):
    global values_0, values_1, values_2, values_3, values_4, values_5, values_6, values_7, values_8
    v0 = values_0[node.index("0")]
    v1 = values_1[node.index("1")]
    v2 = values_2[node.index("2")]
    v3 = values_3[node.index("3")]
    v4 = values_4[node.index("4")]
    v5 = values_5[node.index("5")]
    v6 = values_6[node.index("6")]
    v7 = values_7[node.index("7")]
    v8 = values_8[node.index("8")]
    valorTotal = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
    return valorTotal

def subNodes(node):
    global NodesExpanded
    NodesExpanded = NodesExpanded + 1

    nextPaths = []
    nextPaths.append(PuzzleState(move(node.state, 1), node, 1, node.depth + 1, node.cost + 1, 0))
    nextPaths.append(PuzzleState(move(node.state, 2), node, 2, node.depth + 1, node.cost + 1, 0))
    nextPaths.append(PuzzleState(move(node.state, 3), node, 3, node.depth + 1, node.cost + 1, 0))
    nextPaths.append(PuzzleState(move(node.state, 4), node, 4, node.depth + 1, node.cost + 1, 0))
    nodes = []
    for procPaths in nextPaths:
        if (procPaths.state != None):
            nodes.append(procPaths)
    return nodes

def move(state, direction):
    # generate a copy
    newState = state[:]

    # obtain poss of 0
    index = newState.index(0)

    if (index == 0):
        if (direction == 1):
            return None
        if (direction == 2):
            temp = newState[0]
            newState[0] = newState[3]
            newState[3] = temp
        if (direction == 3):
            return None
        if (direction == 4):
            temp = newState[0]
            newState[0] = newState[1]
            newState[1] = temp
        return newState
    if (index == 1):
        if (direction == 1):
            return None
        if (direction == 2):
            temp = newState[1]
            newState[1] = newState[4]
            newState[4] = temp
        if (direction == 3):
            temp = newState[1]
            newState[1] = newState[0]
            newState[0] = temp
        if (direction == 4):
            temp = newState[1]
            newState[1] = newState[2]
            newState[2] = temp
        return newState
    if (index == 2):
        if (direction == 1):
            return None
        if (direction == 2):
            temp = newState[2]
            newState[2] = newState[5]
            newState[5] = temp
        if (direction == 3):
            temp = newState[2]
            newState[2] = newState[1]
            newState[1] = temp
        if (direction == 4):
            return None
        return newState
    if (index == 3):
        if (direction == 1):
            temp = newState[3]
            newState[3] = newState[0]
            newState[0] = temp
        if (direction == 2):
            temp = newState[3]
            newState[3] = newState[6]
            newState[6] = temp
        if (direction == 3):
            return None
        if (direction == 4):
            temp = newState[3]
            newState[3] = newState[4]
            newState[4] = temp
        return newState
    if (index == 4):
        if (direction == 1):
            temp = newState[4]
            newState[4] = newState[1]
            newState[1] = temp
        if (direction == 2):
            temp = newState[4]
            newState[4] = newState[7]
            newState[7] = temp
        if (direction == 3):
            temp = newState[4]
            newState[4] = newState[3]
            newState[3] = temp
        if (direction == 4):
            temp = newState[4]
            newState[4] = newState[5]
            newState[5] = temp
        return newState
    if (index == 5):
        if (direction == 1):
            temp = newState[5]
            newState[5] = newState[2]
            newState[2] = temp
        if (direction == 2):
            temp = newState[5]
            newState[5] = newState[8]
            newState[8] = temp
        if (direction == 3):
            temp = newState[5]
            newState[5] = newState[4]
            newState[4] = temp
        if (direction == 4):
            return None
        return newState
    if (index == 6):
        if (direction == 1):
            temp = newState[6]
            newState[6] = newState[3]
            newState[3] = temp
        if (direction == 2):
            return None
        if (direction == 3):
            return None
        if (direction == 4):
            temp = newState[6]
            newState[6] = newState[7]
            newState[7] = temp
        return newState
    if (index == 7):
        if (direction == 1):
            temp = newState[7]
            newState[7] = newState[4]
            newState[4] = temp
        if (direction == 2):
            return None
        if (direction == 3):
            temp = newState[7]
            newState[7] = newState[6]
            newState[6] = temp
        if (direction == 4):
            temp = newState[7]
            newState[7] = newState[8]
            newState[8] = temp
        return newState
    if (index == 8):
        if (direction == 1):
            temp = newState[8]
            newState[8] = newState[5]
            newState[5] = temp
        if (direction == 2):
            return None
        if (direction == 3):
            temp = newState[8]
            newState[8] = newState[7]
            newState[7] = temp
        if (direction == 4):
            return None
        return newState


# MAIN**************************************************************
def main():
    global GoalNode

    info = "6,1,8,4,0,2,7,3,5" #20
    info = "8,6,4,2,1,3,5,7,0" #26
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('initialBoard')
    args = parser.parse_args()
    data = args.initialBoard.split(",")

    # Build initial board state
    InitialState = []
    InitialState.append(int(data[0]))
    InitialState.append(int(data[1]))
    InitialState.append(int(data[2]))
    InitialState.append(int(data[3]))
    InitialState.append(int(data[4]))
    InitialState.append(int(data[5]))
    InitialState.append(int(data[6]))
    InitialState.append(int(data[7]))
    InitialState.append(int(data[8]))

    function = args.method
    if (function == "bfs"):
        bfs(InitialState)
    if (function == "dfs"):
        dfs(InitialState)
    if (function == "ast"):
        ast(InitialState)


    # Save total path result
    deep = GoalNode.depth
    moves = []
    while InitialState != GoalNode.state:
        if GoalNode.move == 1:
            path = 'Up'
        if GoalNode.move == 2:
            path = 'Down'
        if GoalNode.move == 3:
            path = 'Left'
        if GoalNode.move == 4:
            path = 'Right'
        moves.insert(0, path)
        GoalNode = GoalNode.parent

    print("hello!")
    print("path: ", moves)
    print("cost: ", len(moves))
    print("nodes expanded: ", str(NodesExpanded))
    print("search_depth: ", str(deep))
    print("MaxSearchDeep: ", str(MaxSearchDeep))
main()
