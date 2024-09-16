import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the transformations for the test dataset (same as for training and validation)
test_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize(256),  # Rescale the shorter side to 256
    transforms.CenterCrop(256),  # Center-crop to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using the given mean and std
])
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the test dataset from a directory
test_dir = 'big_dataset/dataset_8/test_8'  # Replace with the actual path to your test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# Create the DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
# Assuming you have already trained and saved the model earlier using:
# torch.save(model, 'best_alexnet_model_with_architecture.pth')

# Now let's load the saved model with architecture and weights for testing
model = torch.load('best_alexnet_model_with_architecture.pth')  # Load the entire model (architecture + weights)
model = model.to(device)  # Send the model to the device (GPU/CPU)

# Open a file to save the test error rates
with open('test_error_rates.txt', 'w') as file:
    file.write("Test Results:\n")

    # Test the model on test data
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate top-1 and top-5 accuracy
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()

            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            # Top-1 accuracy
            correct_top1 += correct[:1].reshape(-1).float().sum(0).item()

            # Top-5 accuracy
            correct_top5 += correct[:5].reshape(-1).float().sum(0).item()

            total_samples += labels.size(0)

    # Calculate the top-1 and top-5 error rates
    top1_error_rate = 100 - (correct_top1 / total_samples) * 100
    top5_error_rate = 100 - (correct_top5 / total_samples) * 100

    # Print the error rates
    print(f"Test - Top-1 Error Rate: {top1_error_rate}%, Top-5 Error Rate: {top5_error_rate}%")

    # Save the error rates to the text file
    file.write(f"Top-1 Error Rate: {top1_error_rate:.4f}%\n")
    file.write(f"Top-5 Error Rate: {top5_error_rate:.4f}%\n")
