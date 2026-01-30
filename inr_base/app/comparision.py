import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


os.makedirs("result", exist_ok=True)


x = torch.linspace(-3.14, 3.14, 1000).unsqueeze(1)
y_target = torch.sin(5 * x)



# y_target = torch.sin(7.3 * x)

# y_target = torch.sign(torch.sin(3 * x))

# 4. AM
# y_target = torch.sin(3 * x) * torch.cos(0.5 * x)

# y_target = torch.relu(x - 1.0)

# 6. Sawtooth
# y_target = (x % 2*3.14) / 3.14 - 1.0


omega = 1.0

# sin(a * ωx)
class SinWithScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.sin(self.a * omega * x)

# sin(ωx + b)
class SinWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.sin(omega * x + self.b)


def train(model, x, y, steps=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        y_pred = model(x)
        loss = ((y_pred - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model(x).detach()


model1 = SinWithScaling()
model2 = SinWithBias()

y_pred1 = train(model1, x, y_target)
y_pred2 = train(model2, x, y_target)


plt.figure(figsize=(10, 4))
plt.plot(x.numpy(), y_target.numpy(), label="Target: sin(5x)", color="black", linestyle="--")
plt.plot(x.numpy(), y_pred1.numpy(), label="sin(a·ωx)", alpha=0.7)
plt.plot(x.numpy(), y_pred2.numpy(), label="sin(ωx + b)", alpha=0.7)
plt.legend()
plt.title("Comparison of sin(a·ωx) vs sin(ωx + b)")
plt.grid(True)


plt.savefig("result/sin_comparison.png")

mse1 = ((y_pred1 - y_target) ** 2).mean().item()
mse2 = ((y_pred2 - y_target) ** 2).mean().item()

print(f"MSE for sin(a·ωx):     {mse1:.6f}")
print(f"MSE for sin(ωx + b):   {mse2:.6f}")
