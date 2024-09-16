import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define the transformation
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize(256),  # Rescale the shorter side to 256
    transforms.CenterCrop(256),  # Center-crop to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using the given mean and std
])

# Load the training and validation datasets
train_dir = '../big_dataset/dataset_128/augmented_dataset_128'
val_dir = '../big_dataset/dataset_128/val_128'
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Track printed samples for each class
max_prints_per_class = 5

class AlexNet(nn.Module):
    def __init__(self, num_classes=128):  # Changed to 128 classes
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
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),  # Adjusted size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.classifier(x)
        return x

# Helper functions for accuracy calculation
def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """Computes the top-1 and top-5 accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Training and Validation Loop
num_epochs = 30
best_val_loss = float('inf')  # Initialize the best validation loss

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available", flush=True)
        device = torch.device("cuda")
    else:
        print("CUDA not available", flush=True)
        device = torch.device("cpu")

    model = AlexNet(num_classes=128)  # Changed to 128 classes
    print("Finish creating the model", flush=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Open a file to save the error rates
    with open('error_rates.txt', 'w') as file:
        file.write("Epoch\tValidation Loss\tTop-1 Error Rate\tTop-5 Error Rate\n")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print training statistics
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader)}", flush=True)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_top1 = 0
            correct_top5 = 0
            total_samples = 0

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # Calculate the loss (error) during validation
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate top-1 and top-5 accuracy
                    _, pred = outputs.topk(5, 1, True, True)
                    pred = pred.t()

                    correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    # Top-1 accuracy
                    correct_top1 += correct[:1].reshape(-1).float().sum(0).item()

                    # Top-5 accuracy
                    correct_top5 += correct[:5].reshape(-1).float().sum(0).item()

                    total_samples += labels.size(0)

            # Calculate average validation loss
            val_loss /= len(val_loader)

            # Check if this is the best validation loss and save the model if it is
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, 'best_alexnet_model_with_architecture.pth')  # Save the model
                print(f"Saved model with validation loss: {val_loss}", flush=True)

            # Calculate top-1 and top-5 error rates as percentages
            top1_error_rate = 100 - (correct_top1 / total_samples) * 100
            top5_error_rate = 100 - (correct_top5 / total_samples) * 100

            # Print error rates
            print(f"Validation Loss: {val_loss}, Top-1 Error Rate: {top1_error_rate}%, Top-5 Error Rate: {top5_error_rate}%", flush=True)

            # Save error rates to the text file and flush the buffer
            file.write(f"{epoch + 1}\t{val_loss:.4f}\t{top1_error_rate:.4f}\t{top5_error_rate:.4f}\n")
            file.flush()  # Ensure content is written to file after every epoch
