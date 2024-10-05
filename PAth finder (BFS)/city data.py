class city:
    def bfs_shortest_path(self,graph, start, goal):
        queue = [start]
        distance = {start: 0}
        parent = {start: None}

        while queue:
            current = queue.pop(0)
            if current==None:
                break
            if current == goal:
                break
            for neighbor in graph[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
        if goal not in distance:
            return None
        path = []
        while goal is not None:
            path.append(goal)
            goal = parent[goal]
        return path
with open(file="pb.txt.txt") as file:
    data=file.readlines()

my_dict={}
new_dict={}
for i in data:
    k=i.split(",")
    p=k[0].split(" ")
    # print(p)
    new_dict[p[1]]=p[0]
    l=k[1].split(" ")
    l=l[1:len(l)-1]
    m=[]
    m.extend(l)
    for i in m:
        if i=="":
            m.remove(i)
    index=p[0]
    my_dict[index]=m
print(my_dict)
print(new_dict)
f=input("Enter the first city: ")
f2=input("Enter the second city: ")
first=None
second=None
for key in new_dict:
    if key==f:
        first=new_dict[key]
    elif key==f2:
        second=new_dict[key]
c=city()
path = c.bfs_shortest_path(my_dict, first, second)
if path==None:
    print("no such path found!")
else:
    for i in range(len(path)-1,-1,-1):
        value=path[i]
        for key in new_dict:
            if new_dict[key]==str(value):
                print(f"--->{key}",end="")
                break
