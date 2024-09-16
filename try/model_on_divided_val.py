import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Normalize using the given mean and std
])

# Load the training and validation datasets
train_dir = '../big_dataset/train'
val_dir = '../big_dataset/divided_val'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
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
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),  # Updated to 12544
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            # nn.Softmax(dim=1)  # Remove this line
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.classifier(x)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """Computes the top-1 and top-5 accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Changed to reshape
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Training loop
num_epochs = 10

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
    else:
        print("CUDA not available")
        device = torch.device("cpu")

    model = AlexNet(num_classes=1000)
    print("Model created.")

    # Apply the weight initialization function
    model.apply(initialize_weights)
    print("Weights initialized.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed = min((batch_idx + 1) * len(inputs), len(train_loader.dataset))
            print(f'Train Epoch: {epoch} [{processed}/{len(train_loader.dataset)}'
                  f' ({100. * processed / len(train_loader.dataset):.0f}%)]\tBatch: {batch_idx + 1}\tLoss: {loss.item():.6f}')

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        top1_acc = 0
        top5_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                top1, top5 = calculate_topk_accuracy(outputs, labels)
                top1_acc += top1.item()
                top5_acc += top5.item()

                # Print some sample outputs for the current epoch
                if i < 5:  # Print only for first 5 batches
                    for j in range(inputs.size(0)):
                        print(f"Image: {val_dataset.imgs[i * inputs.size(0) + j][0]}, "
                              f"True Label: {labels[j].item()}, "
                              f"Predicted Label: {outputs[j].topk(1)[1].item()}")

        # Calculate average accuracy
        top1_acc /= len(val_loader)
        top5_acc /= len(val_loader)
        print(f"Top-1 Accuracy: {top1_acc}%, Top-5 Accuracy: {top5_acc}%")
