import numpy as np
import random

num_ball = 18
width = 30
max_velo = 2

position1 = np.random.rand(num_ball)
position1 = position1 * (width/2-5)
position2 = np.random.rand(num_ball)
position2 = position2 * (width/2-5) + width/2 + 5
position = np.append(position1, position2)
np.random.shuffle(position)
velo = np.random.rand(num_ball*2)
velo = velo * max_velo


for i in range(num_ball):
    text1 = "ba-loc(b" + str(i+1) + ", x) = " + str(max(0, round(position[i*2]))) + ";"
    text2 = "ba-loc(b" + str(i+1) + ", y) = " + str(max(0, round(position[i*2+1]))) + ";"
    print(text1)
    print(text2)

for i in range(num_ball):
    text1 = "ba-velo(b" + str(i+1) + ", x) = " + str(round(velo[i*2] * random.choice([-1, 1])))+ ";"
    text2 = "ba-velo(b" + str(i+1) + ", y) = " + str(round(velo[i*2+1] * random.choice([-1, 1])))+ ";"
    print(text1)
    print(text2)

'''for i in range(num_ball):
    text1 = "ba_x_loc(@b" + str(i+1) + ", x" + str(max(1, round(position[i*2]))) + ");"
    text2 = "ba_y_loc(@b" + str(i+1) + ", y" + str(max(1, round(position[i*2+1])))+ ");"
    print(text1)
    print(text2)

for i in range(num_ball):
    text1 = "ba_x_velo(@b" + str(i+1) + ") = @"
    a = round(velo[i*2] * random.choice([-1, 1]))
    if a < 0:
        text1 += 'n' + str(-a) + ';'
    else:
        text1 += str(a) + ';'
    text2 = "ba_y_velo(@b" + str(i+1) + ") = @"
    b = round(velo[i*2+1] * random.choice([-1, 1]))
    if b < 0:
        text2 += 'n' + str(-b) + ';'
    else:
        text2 += str(b) + ';'
    print(text1)
    print(text2)'''

'''
for i in range(width):
    print("FROM_TO_x_1(" + 'x' + str(i+1) + ', x' + str(min(width, i+2)) + ');')
for i in range(width):
    print("FROM_TO_x_2(" + 'x' + str(i+1) + ', x' + str(min(width, i+3)) + ');')
for i in range(width):
    print("FROM_TO_x_3(" + 'x' + str(i+1) + ', x' + str(min(width, i+4)) + ');')
for i in range(width):
    print("FROM_TO_x_n1(" + 'x' + str(i+1) + ', x' + str(max(1, i)) + ');')
for i in range(width):
    print("FROM_TO_x_n2(" + 'x' + str(i+1) + ', x' + str(max(1, i-1)) + ');')
for i in range(width):
    print("FROM_TO_x_n3(" + 'x' + str(i+1) + ', x' + str(max(1, i-2)) + ');')
for i in range(width):
    print("FROM_TO_x_0(" + 'x' + str(i+1) + ', x' + str(i+1) + ');')

x = ''
for i in range(width):
    x = x + 'x' + str(i+1) + ','
print(x)
y = ''
for i in range(width):
    y = y + 'y' + str(i+1) + ','
print(y)

x = ''
for i in range(3):
    x = x + 'v' + str(i+1) + ','
    x = x + 'v' + 'n'+str(i+1) + ','
print(x)
#for i in range(9):
#    print("ba_x_velo(" + 'b' + str(i+1)  + ')' + '=' + str(1) + ';')

for i in range(9):
    print("ba_x_loc(" + 'b' + str(i+1)  +', x'+ str(round(position[i * 2])) +');')

for i in range(9):
    print("ba_y_loc(" + 'b' + str(i+1)  +', y'+ str(round(position[i * 2])) +');')
for i in range(25):
    print('VX(' + 'x' + str(i+1)+ ') = ' + str(i+1) + ';')
for i in range(25):
    print('VY(' + 'y' + str(i+1)+ ') = ' + str(i+1) + ';')
for i in range(25):
    print('NEIGHBORY(' + 'y' + str(min(25, i+1)) + ', y' + str(min(25, i+1)) + ');')
    print('NEIGHBORY(' + 'y' + str(min(25, i+1)) + ', y' + str(min(25, i+2)) + ');')
    print('NEIGHBORY(' + 'y' + str(min(25, i+2)) + ', y' + str(min(25, i+1)) + ');')

for i in range(width):
    print('VX(' + 'x' + str(i+1)+ ') = ' + str(i+1) + ';')
for i in range(width):
    print('VY(' + 'y' + str(i+1)+ ') = ' + str(i+1) + ';')
'''
