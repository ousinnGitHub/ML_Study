import numpy as np


def sigmoid(x, backword=False):
    if backword == True:
        return x * (1 - x)
    return 1.0 / (1 + np.exp(-x))


x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]])

print(x.shape)

y = np.array([[0], [1], [1], [0], [0]])
print(y.shape)

np.random.seed(1)
w0 = 2 * np.random.rand(3, 4) - 1
w1 = 2 * np.random.rand(4, 1) - 1
print(w0)
print(w1)

for j in range(6000):
    l0 = x
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = l2 - y
    if(j % 1000) == 0:
        print("Error:", np.mean(np.abs(l2_error)))

    l2_delta = l2_error * sigmoid(l2, backword=True)
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, backword=True)

    w1 -= l1.T.dot(l2_delta)
    w0 -= l0.T.dot(l1_delta)
