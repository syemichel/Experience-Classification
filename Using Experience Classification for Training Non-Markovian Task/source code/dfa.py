import math
from collections import defaultdict

def dfs(graph, node, visited, path, all_paths, end_node):
    visited[node] = True
    path.append(node)

    if node == end_node:
        all_paths.append(path.copy())
    elif node in graph:
        for child in graph[node]:
            if not visited[child]:
                dfs(graph, child, visited, path, all_paths, end_node)

    path.pop()
    visited[node] = False

def find_all_paths(graph, start_node, end_node):
    visited = defaultdict(bool)
    path = []
    all_paths = []

    dfs(graph, start_node, visited, path, all_paths, end_node)

    return all_paths

def calculate_avg_distance(graph, end_node):
    avg_distances = {}

    for start_node in graph:
        all_paths = find_all_paths(graph, start_node, end_node)
        total_distance = sum([len(path) - 1 for path in all_paths])
        avg_distance = total_distance / len(all_paths) if all_paths else float('inf')
        avg_distances[start_node] = avg_distance

    return avg_distances

graph = {
    '1': ['1', '2', '3', '4', '5'],
    '2': ['2'],
    '3': ['3', '2', '6', '5', '7'],
    '4': ['4', '2', '5', '8', '9'],
    '5': ['5', '2', '7', '9', '10'],
    '6': ['6', '11', '2', '7', '12'],
    '7': ['7', '12', '2', '10', '13'],
    '8': ['8', '2', '9', '14', '15'],
    '9': ['9', '2', '10', '15', '16'],
    '10': ['10', '13', '2', '16', '17'],
    '11': ['11', '2', '12'],
    '12': ['2', '13'],
    '13': ['13', '17', '2'],
    '14': ['14', '2', '15'],
    '15': ['15', '2', '16'],
    '16': ['16', '17', '2'],
    '17': ['17', '2']

}

end_node = '17'
avg_distances = calculate_avg_distance(graph, end_node)
sorted_avg_distances = sorted(avg_distances.items(), key=lambda x: x[1])

print(f"节点到终端节点{end_node}的平均距离：")
for node, distance in sorted_avg_distances:
    print(f"{node}: {distance}")
N = 8
L_min = sorted_avg_distances[1][1]
L_max = sorted_avg_distances[15][1]
print(L_max)
print(L_min)
for node, distance in sorted_avg_distances:
    if node == '17':
        print(f"{node}: {N - 1}")
    elif node == '2':
        print(f"{node}: {N - 2}")
    else:
        print(f"{node}: {round(N - 3 - (distance - L_min) / (L_max - L_min) * (N-3))}")