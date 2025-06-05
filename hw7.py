from micrograd.engine import Value  # 需要存好 micrograd 的 Value 类

x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

learning_rate = 0.1
for epoch in range(50):
    # Forward
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    x.grad = 0
    y.grad = 0
    z.grad = 0
    f.backward()

    # Gradient Descent
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: f = {f.data:.4f}, x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")
