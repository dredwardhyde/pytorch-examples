import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms)
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False, pin_memory=True)
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss().to(device=device)
n_epochs = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


model = Net().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device=device), labels.to(device=device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print("Training accuracy: %f" % (correct / total))


with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in val_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print("Validation accuracy: %f" % (correct / total))
