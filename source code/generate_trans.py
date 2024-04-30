import re
from classify1 import *
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

# Example usage
file_path = 'dot.txt'
trans_express = 2
with open(file_path, 'r') as file:
    lines = [line.strip() for line in file]
classified_result = classify_lines(lines)
# To print or use the classified result
if trans_express == 2:
    converted_transitions = [convert_transition_with_regex(transition) for transition in lines]
    # 输出结果
    for converted in converted_transitions:
        print(converted)
graph = {}
for destination, lines in classified_result.items():
    trans = []
    for line in lines:
        new_tuple = generate_transitions(line)
        if str(new_tuple[0]) not in graph:
            graph[str(new_tuple[0])] = []
        graph[str(new_tuple[0])].append(str(new_tuple[1]))
        trans.append(new_tuple)
    if trans_express == 1:
        print(generate_expression(trans))
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

}'''
classify_dfa_states(graph, accepted_state='145', error_state='', num=14)
generate_reward_function(num=14, interval=7)


