import math
import re
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

def classify_lines(lines):
    classification_result = {}
    for line in lines:
        # Split the line based on '->' to find the target number
        parts = line.split("->")
        if len(parts) == 2:
            destination = parts[1].split("[")[0].strip()
            if destination not in classification_result:
                classification_result[destination] = []
            classification_result[destination].append(line)
    return classification_result

def generate_expression(transitions):
    # 初始化一个空字典，用于存储各个目标状态的转移条件
    expressions = {}

    # 遍历每个转移，转换条件中的 '&' 为 '^'
    for src, dst, condition in transitions:
        # 替换 '&' 为 '^'
        condition = condition.replace("&", "^")
        # 为每个目标状态构建表达式
        if dst in expressions:
            expressions[dst].append(f"(q{src} ^ ({condition}))")
        else:
            expressions[dst] = [f"(q{src} ^ ({condition}))"]

    # 构建最终的表达式字符串
    result_expressions = []
    for dst, conds in expressions.items():
        result_expressions.append(f"q{dst}' = {' | '.join(conds)};")

    return '\n'.join(result_expressions)

def generate_transitions(input_transition):
    pattern = r"(\d+) -> (\d+) \[label=\"([^\"]+)\"\]"
    match = re.search(pattern, input_transition)
    if match:
        transition_tuple = (int(match.group(1)), int(match.group(2)), match.group(3))
    else:
        transition_tuple = None
    return transition_tuple

def generate_reward_function(num=6, max_reward=100, interval=10):
    reward_str = ''
    for i in range(num-2):
        reward_str += f"{interval*i}*[as==@{i+1}] + "
    reward_str += f"{max_reward}*[as==@{num}] "
    for i in range(num-2):
        reward_str += f"- {interval*i}*[pas==@{i+1}] "
    reward_str += f"- {max_reward}*[pas==@{num}] "
    print(reward_str)

def convert_transition_with_regex(input_line):
    # 使用正则表达式来匹配输入行的模式
    match = re.match(r'(\d+) -> (\d+) \[label="([^"]+)"\];', input_line)

    # 如果匹配成功，则提取相应的部分
    if match:
        from_state, to_state, condition = match.groups()
        # 根据提取的值构造新的字符串
        return f"else if(pds==@q{from_state} ^ ({condition})) then @q{to_state}"
    else:
        # 如果匹配不成功，返回错误消息
        return "Invalid input format"
