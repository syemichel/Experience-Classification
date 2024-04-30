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

def calculate_per_node(graph, start_node, end_node, value):
    next_nodes = []
    distance = 0

    if start_node == end_node:
        return -1
    for node in graph[start_node]:
        if node != start_node:
            next_nodes.append(node)

    path_num = len(next_nodes)
    if next_nodes == []:
        value[start_node] = 0
        return 0
    for node in next_nodes:
        if node == end_node:
            distance += 1
        else:
            if node in value:
                dist = value[node]
            else:
                dist = calculate_per_node(graph, node, end_node, value)
            if dist == 0:
                path_num -= 1
            else:
                distance += (dist + 1)
    # print(start_node + ': ' + str(distance / path_num))
    value[start_node] = distance / path_num
    return distance / path_num
def calculate_avg_distance(graph, end_node):
    avg_distances = {}
    value = {}
    for start_node in graph:
        # print(start_node)
        dist = calculate_per_node(graph, start_node, end_node, value)
        if dist == 0:
            dist = float('inf')
        elif dist == -1:
            dist = 0
        avg_distances[start_node] = dist
    return avg_distances

def classify_dfa_states(graph, accepted_state='17', error_state='2', num=8):
    end_node = accepted_state
    avg_distances = calculate_avg_distance(graph, end_node)
    sorted_avg_distances = sorted(avg_distances.items(), key=lambda x: x[1])
    L_min = 999999
    L_max = 0
    print(f"节点到终端节点{end_node}的平均距离：")
    for node, distance in sorted_avg_distances:
        print(f"{node}: {distance}")
        if distance < L_min and distance > 0:
            L_min = distance
        if distance > L_max and distance > 0 and distance < 999999:
            L_max = distance
    N = num
    print('max:',L_max)
    print('min:',L_min)
    classification = {}
    for i in range(N):
        classification[str(i+1)] = ''
    for node, distance in sorted_avg_distances:
        if node == accepted_state:
            classification[f"{N - 1 + 1}"] += (f"ds==@q{node} | ")
            print(f"{node}: {N - 1 + 1}")
        elif node == error_state:
            classification[f"{N - 2 + 1}"] += (f"ds==@q{node} | ")
            print(f"{node}: {N - 2 + 1}")
        else:
            classification[f"{round(N - 3 - (distance - L_min) / (L_max - L_min) * (N-3)) + 1}"] += (f"ds==@q{node} | ")
            print(f"{node}: {round(N - 3 - (distance - L_min) / (L_max - L_min) * (N-3)) + 1}")

    for key, value in classification.items():
        print('else if(' + value[:-3] + ') then '+ f"@{key}")

'''graph = {
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

print(calculate_avg_distance(graph, '17'))'''

