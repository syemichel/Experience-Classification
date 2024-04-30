import re
from classify1 import *
# Example usage
file_path = 'dot.txt'  # write in transition information for dfa in the dot file
trans_express = 2
# We achieve two conversion formats, defaulting to the second one, your can use the first on by changing it to 1
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

classify_dfa_states(graph, accepted_state='145', error_state='', num=14)
# set the accepted_state and the classification number
generate_reward_function(num=14, interval=7)
# generate reward functions


