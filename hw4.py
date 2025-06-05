# 本作业使用 Python 实作一个多层感知器（MLP），以七段显示器输入向量（共7位）作为输入，输出对应数字的4位二进制表示。
包含一个隐藏层（6个节点），激活函数为 Sigmoid，损失函数为 MSE，并使用数值梯度下降法手动优化参数。

import numpy as np

# === 函数定义 ===

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  
    s = sigmoid(x)
    return s * (1 - s)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def compute_numerical_gradient(loss_fn, params, epsilon=1e-5):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        original = params[i]
        
        params[i] = original + epsilon
        loss1 = loss_fn(params)
        
        params[i] = original - epsilon
        loss2 = loss_fn(params)
        
        grad[i] = (loss1 - loss2) / (2 * epsilon)
        params[i] = original 
    return grad

# 七段输入
segments = {
    0: (1,1,1,1,1,1,0),
    1: (0,1,1,0,0,0,0),
    2: (1,1,0,1,1,0,1),
    3: (1,1,1,1,0,0,1),
    4: (0,1,1,0,0,1,1),
    5: (1,0,1,1,0,1,1),
    6: (1,0,1,1,1,1,1),
    7: (1,1,1,0,0,0,0),
    8: (1,1,1,1,1,1,1),
    9: (1,1,1,1,0,1,1),
}

binary_outputs = {
    0: (0,0,0,0),
    1: (0,0,0,1),
    2: (0,0,1,0),
    3: (0,0,1,1),
    4: (0,1,0,0),
    5: (0,1,0,1),
    6: (0,1,1,0),
    7: (0,1,1,1),
    8: (1,0,0,0),
    9: (1,0,0,1),
}

X = np.array([segments[i] for i in range(10)]) 
Y = np.array([binary_outputs[i] for i in range(10)]) 

input_size = 7
hidden_size = 6
output_size = 4


W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

def pack_parameters(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

def unpack_parameters(params):
    i, h, o = input_size, hidden_size, output_size
    idx = 0
    W1 = params[idx:idx+i*h].reshape(i, h)
    idx += i * h
    b1 = params[idx:idx+h].reshape(1, h)
    idx += h
    W2 = params[idx:idx+h*o].reshape(h, o)
    idx += h * o
    b2 = params[idx:idx+o].reshape(1, o)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1  # 10 x hidden
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2  # 10 x output
    a2 = sigmoid(z2)
    return a1, a2

def loss_with_params(params):
    W1_, b1_, W2_, b2_ = unpack_parameters(params)
    _, y_pred = forward(X, W1_, b1_, W2_, b2_)
    return mse(y_pred, Y)

params = pack_parameters(W1, b1, W2, b2)

learning_rate = 0.5
epochs = 3000

for epoch in range(epochs):
    grad = compute_numerical_gradient(loss_with_params, params)
    params -= learning_rate * grad

    if epoch % 200 == 0:
        loss = loss_with_params(params)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

W1, b1, W2, b2 = unpack_parameters(params)
_, predictions = forward(X, W1, b1, W2, b2)

print("\n=== 测试结果 ===")
for i in range(10):
    binary_str = "".join(map(str, Y[i]))
    pred_str = "".join(map(str, np.round(predictions[i]).astype(int)))
    print(f"Input {X[i]} → Predict: {pred_str} | Expect: {binary_str}")

