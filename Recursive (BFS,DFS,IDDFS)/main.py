class Graph:
    def __init__(self, gdict=None):
        if gdict is None:
            gdict = {}
        self.gdict = gdict

    def addEdge(self, vertex, edge):
        self.gdict[vertex].append(edge)
    def bfs_recursive(self, queue, visited):
        if not queue:
            return
        vertex = queue.pop(0)
        print(vertex,end=" ")
        for adjacent in self.gdict[vertex]:
            if adjacent not in visited:
                visited.append(adjacent)
                queue.append(adjacent)
        self.bfs_recursive(queue, visited)
    def bfs(self, vertex):
        visited = [vertex]
        queue = [vertex]
        self.bfs_recursive(queue, visited)
    def dfs_recursive(self, vertex, visited):
        print(vertex,end="")
        visited.append(vertex)
        for adjacent in self.gdict[vertex]:
            if adjacent not in visited:
                self.dfs_recursive(adjacent, visited)
    def dfs(self, vertex):
        visited = []
        self.dfs_recursive(vertex, visited)

customDict = { "a" : ["b","c"],
            "b" : ["a", "d", "e"],
            "c" : ["a", "e"],
            "d" : ["b", "e", "f"],
            "e" : ["d", "f", "c"],
            "f" : ["d", "e"]
               }
g = Graph(customDict)
# g.dfs("a")
# g.bfs("f")


# Depth Limited Search (DLS)
def depth_limited_search(node, goal, depth):
    if node == goal:
        return True
    if depth <= 0:
        return False
    for neighbor in graph.get(node, []):
        if depth_limited_search(neighbor, goal, depth - 1):
            return True
    return False

# Iterative Deepening DFS (IDDFS)
def iddfs(start, goal, max_depth):
    for depth in range(max_depth):
        print(f"Depth level: {depth}")
        if depth_limited_search(start, goal, depth):
            return True
    return False

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G', 'H'],
    'F': [],
    'G': [],
    'H': []
}

# Example usage
start_node = 'A'
goal_node = 'G'
max_depth_limit = 5

if iddfs(start_node, goal_node, max_depth_limit):
    print(f"Goal node {goal_node} found!")
else:
    print(f"Goal node {goal_node} not found within depth limit {max_depth_limit}.")