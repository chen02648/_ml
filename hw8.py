import torch

x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1

for epoch in range(50):
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    f.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        z -= learning_rate * z.grad

        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: f = {f.item():.4f}, x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")
