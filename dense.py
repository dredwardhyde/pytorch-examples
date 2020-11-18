from collections import OrderedDict

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

torch.set_printoptions(edgeitems=2, linewidth=75)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c, device=device).unsqueeze(1)
t_u = torch.tensor(t_u, device=device).unsqueeze(1)
n_epochs = 5000
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
loss_fn = nn.MSELoss()
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]
t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]
t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val

model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
])).to(device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(1, n_epochs + 1):
    t_p_train = model(t_un_train)
    loss_train = loss_fn(t_p_train, t_c_train)
    t_p_val = model(t_un_val)
    loss_val = loss_fn(t_p_val, t_c_val)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    if epoch == 1 or epoch % 1000 == 0:
        print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
              f" Validation loss {loss_val.item():.4f}")

t_range = torch.arange(20., 90., device=device).unsqueeze(1)
fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.cpu().numpy(), t_c.cpu().numpy(), 'o')
plt.plot(t_range.cpu().numpy(), model(0.1 * t_range).cpu().detach().numpy(), 'c-')
plt.plot(t_u.cpu().numpy(), model(0.1 * t_u).cpu().detach().numpy(), 'kx')
plt.savefig('function.png')
