import matplotlib.pyplot as plt
import csv
import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp((-1) * x))


# 1
x1 = []
x2 = []
label = []
color = []
# reading data from csv file
with open('dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    lineCount = 0
    for row in reader:
        if lineCount != 0 :
            x1.append(float(row[0]))
            x2.append(float(row[1]))
            if int(row[2]) == 0:
                color.append('blue')
                label.append(0)
            elif int(row[2]) == 1:
                color.append('red')
                label.append(1)
        else:
            lineCount = 1

# plot all of data
plt.scatter(x1, x2, None, color)
plt.show()

# dividing data to training and test set
train = []
for i in range(150):
    train.append([x1[i], x2[i], label[i]])
test = []
for i in range(150, 180):
    test.append((x1[i], x2[i], label[i]))




# 2,3,4(implementing one perceptron model)
w = [random.random(), random.random()]
b = random.random()
number_steps = 5000
lr = 1
n = 150
for i in range(number_steps):
    grad = [0, 0, 0]
    for j in range(n):
        tmp = (train[j][0] * w[0]) + (train[j][1] * w[1]) + b
        res = 1 / (1 + math.exp((-1) * tmp))
        grad[0] += (-1) * ((float(train[j][2]) / res) + ((float(train[j][2]) - 1) / (1 - res))) * (train[j][0] * res * (1-res))
        grad[1] += (-1) * ((float(train[j][2]) / res) + ((float(train[j][2]) - 1) / (1 - res))) * (train[j][1] * res * (1-res))
        grad[2] += (-1) * ((float(train[j][2]) / res) + ((float(train[j][2]) - 1) / (1 - res))) * (res * (1-res))
    w[0] = w[0] - (lr * grad[0]) / n
    w[1] = w[1] - (lr * grad[1]) / n
    b = b - (lr * grad[2]) / n

color = []
x1Show = []
x2Show = []
accuracy = 0
for i in range(30):
    x1Show.append(test[i][0])
    x2Show.append(test[i][1])
    res = (test[i][0] * w[0]) + (test[i][1] * w[1]) + b
    if res < 0:
        color.append('blue')
    else:
        color.append('red')
    if (res < 0 and test[i][2] == 0) or (res > 0 and test[i][2] == 1):
        accuracy += 1
plt.scatter(x1Show, x2Show, None, color)
plt.show()
print("accuracy = ", float(accuracy)/30)






# 5(implementing model containing three perceptron)
w = [random.random(), random.random()]
v = [random.random(), random.random()]
u = [random.random(), random.random()]
b = [random.random(), random.random(), random.random()]
number_steps = 25000
lr = 3
n = 150
for i in range(number_steps):
    for j in range(n):
        gradw = [0, 0]
        gradv = [0, 0]
        gradu = [0, 0]
        gradb = [0, 0, 0]
        z0 = sigmoid((train[j][0] * w[0]) + (train[j][1] * w[1]) + b[0])
        z1 = sigmoid((train[j][0] * v[0]) + (train[j][1] * v[1]) + b[1])
        y = sigmoid((z0 * u[0]) + (z1 * u[1]) + b[2])
        gradw[0] = 2 * (y - train[j][2]) * y * (1 - y) * u[0] *z0 * (1 - z0) * train[j][0]
        gradw[1] = 2 * (y - train[j][2]) * y * (1 - y) * u[0] *z0 * (1 - z0) * train[j][1]
        gradb[0] = 2 * (y - train[j][2]) * y * (1 - y) * u[0] *z0 * (1 - z0)
        gradv[0] = 2 * (y - train[j][2]) * y * (1 - y) * u[1] * z1 * (1 - z1) * train[j][0]
        gradv[1] = 2 * (y - train[j][2]) * y * (1 - y) * u[1] * z1 * (1 - z1) * train[j][1]
        gradb[1] = 2 * (y - train[j][2]) * y * (1 - y) * u[1] * z1 * (1 - z1)
        gradu[0] = 2 * (y - train[j][2]) * y * (1 - y) * z0
        gradu[1] = 2 * (y - train[j][2]) * y * (1 - y) * z1
        gradb[2] = 2 * (y - train[j][2]) * y * (1 - y)
        w[0] = w[0] - (lr * gradw[0]) / n
        w[1] = w[1] - (lr * gradw[1]) / n
        b[0] = b[0] - (lr * gradb[0]) / n
        v[0] = v[0] - (lr * gradv[0]) / n
        v[1] = v[1] - (lr * gradv[1]) / n
        b[1] = b[1] - (lr * gradb[1]) / n
        u[0] = u[0] - (lr * gradu[0]) / n
        u[1] = u[1] - (lr * gradu[1]) / n
        b[2] = b[2] - (lr * gradb[2]) / n

color = []
x1Show = []
x2Show = []
accuracy = 0
for i in range(30):
    x1Show.append(test[i][0])
    x2Show.append(test[i][1])
    z0 = sigmoid((test[i][0] * w[0]) + (test[i][1] * w[1]) + b[0])
    z1 = sigmoid((test[i][0] * v[0]) + (test[i][1] * v[1]) + b[1])
    res = sigmoid((z0 * u[0]) + (z1 * u[1]) + b[2])
    if res < 0.5:
        color.append('blue')
    else:
        color.append('red')
    if (res < 0.5 and test[i][2] == 0) or (res > 0.5 and test[i][2] == 1):
        accuracy += 1
plt.scatter(x1Show, x2Show, None, color)
plt.show()
print("accuracy = ", float(accuracy)/30)