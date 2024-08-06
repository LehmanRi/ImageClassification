import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define the CNN architecture as per the paper
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Check for GPU availability
if not torch.cuda.is_available():
    print("No GPU available. Exiting the program.")
    exit()

# Set up the model, loss function, and optimizer
device = torch.device("cuda")
model = AlexNet(num_classes=1000).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# Define data loaders for training and validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.FakeData(transform=transform)  # Replace with your dataset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

val_dataset = datasets.FakeData(transform=transform)  # Replace with your dataset
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)


# Function to simulate two GPU usage on one GPU
def simulate_two_gpus(model, data, target):
    # Split the batch into two halves
    half_size = data.size(0) // 2
    data1, data2 = data[:half_size], data[half_size:]
    target1, target2 = target[:half_size], target[half_size:]

    # Forward pass for the first half
    output1 = model(data1)
    loss1 = criterion(output1, target1)
    loss1.backward()

    # Forward pass for the second half
    output2 = model(data2)
    loss2 = criterion(output2, target2)
    loss2.backward()

    # Combine the losses
    total_loss = (loss1 + loss2) / 2

    return total_loss


# Training and validation loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Simulate two GPU usage on one GPU
        loss = simulate_two_gpus(model, data, target)

        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            # Top-1 accuracy
            pred1 = output.argmax(dim=1, keepdim=True)
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()

            # Top-5 accuracy
            _, pred5 = output.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct5 += pred5.eq(target.view(1, -1).expand_as(pred5)).sum().item()

    val_loss /= len(val_loader.dataset)
    total = len(val_loader.dataset)
    top1_acc = 100. * correct1 / total
    top5_acc = 100. * correct5 / total

    print(f'\nValidation set: Average loss: {val_loss:.4f}, '
          f'Top-1 Accuracy: {correct1}/{total} ({top1_acc:.0f}%), '
          f'Top-5 Accuracy: {correct5}/{total} ({top5_acc:.0f}%)\n')


# Run the training and validation
for epoch in range(1, 11):
    train(model, train_loader, criterion, optimizer, epoch)
    validate(model, val_loader, criterion)
