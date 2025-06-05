# 使用chatgpt
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.linspace(-1, 1, 20).unsqueeze(1) 
y = 2 * x + 3 + 0.1 * torch.randn_like(x)  

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.1
epochs = 100

for epoch in range(epochs):
    # Forward
    y_pred = x * w + b
    loss = F.mse_loss(y_pred, y) 

    # Backward
    loss.backward()

    # Gradient Descent
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
