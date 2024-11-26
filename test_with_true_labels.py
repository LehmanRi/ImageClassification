import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import scipy.io
import os


# Define the AlexNet class (ensure this matches the model used during training)
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # Adjust number of classes as needed
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
            nn.Linear(256 * 6 * 6, 4096),  # Adjust input size based on output of features
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Function to extract 5 patches and their reflections
def extract_patches(image, patch_size=224):
    width, height = image.size
    patches = [
        transforms.CenterCrop(patch_size)(image),
        image.crop((0, 0, patch_size, patch_size)),  # top-left
        image.crop((width - patch_size, 0, width, patch_size)),  # top-right
        image.crop((0, height - patch_size, patch_size, height)),  # bottom-left
        image.crop((width - patch_size, height - patch_size, width, height)),  # bottom-right
    ]
    patches += [transforms.functional.hflip(patch) for patch in patches]
    return patches


# Define patch transformations
patch_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load WNID mapping from meta.mat
meta = scipy.io.loadmat('dataset/meta.mat')
number_to_wnid = {int(row[0]): row[1][0] for row in meta['meta']}

# Load true labels from ILSVRC2010_test_ground_truth.txt
with open('ILSVRC2010_test_ground_truth.txt', 'r') as f:
    true_labels = [int(line.strip()) for line in f]

# Path to the folder containing the subfolders for each class
train_data_path = "dataset/train_data"

# Get the list of subfolder names (class names) in the order they appear
class_names = os.listdir(train_data_path)

# Create the dictionary mapping indices to class names
class_to_wnid = {i: class_name for i, class_name in enumerate(class_names)}

if __name__ == '__main__':
    # Load the model and set to evaluation mode
    model = AlexNet(num_classes=1000)
    model.load_state_dict(torch.load('best_alexnet_weights.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Directory containing test images
    test_dir = 'test_data'

    # Open files to save the results
    with open('test_error_rates.txt', 'w') as error_file, open('classification_results.txt', 'w') as class_file:
        error_file.write("Test Results:\n")
        class_file.write("Image Classification Results:\n")

        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        # Iterate over test images
        for idx, image_name in enumerate(os.listdir(test_dir)):
            if image_name.endswith('.JPEG'):
                image_path = os.path.join(test_dir, image_name)

                # Get the true WNID from the ground truth file and meta.mat
                true_label_number = true_labels[idx]
                true_wnid = number_to_wnid[true_label_number]

                # Open image and extract patches
                pil_image = Image.open(image_path).convert('RGB')
                patches = extract_patches(pil_image)

                # Store predictions for all patches
                avg_outputs = torch.zeros(1, len(class_to_wnid)).to(device)

                # Classify each patch and average outputs
                for patch in patches:
                    patch_tensor = patch_transform(patch).unsqueeze(0).to(device)
                    outputs = model(patch_tensor)
                    avg_outputs += outputs

                avg_outputs /= len(patches)

                # Calculate top-1 and top-5 accuracy
                _, pred = avg_outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(
                    torch.tensor([list(class_to_wnid.keys())[list(class_to_wnid.values()).index(true_wnid)]]).to(
                        device).view(1, -1).expand_as(pred))

                correct_top1 += correct[:1].reshape(-1).float().sum(0).item()
                correct_top5 += correct[:5].reshape(-1).float().sum(0).item()
                total_samples += 1

                # Get the top-1 predicted class and confidence
                top1_pred_class = pred[0, 0].item()
                predicted_wnid = class_to_wnid[top1_pred_class]
                top1_confidence = avg_outputs[0, top1_pred_class].item()

                # Save results for each image
                class_file.write(
                    f"{image_name}, True WNID: {true_wnid}, Predicted WNID: {predicted_wnid}, Confidence: {top1_confidence:.4f}\n")

        # Calculate error rates
        top1_error_rate = 100 - (correct_top1 / total_samples) * 100
        top5_error_rate = 100 - (correct_top5 / total_samples) * 100
        print(f"Top-1 Error Rate: {top1_error_rate:.4f}%, Top-5 Error Rate: {top5_error_rate:.4f}%")
        error_file.write(f"Top-1 Error Rate: {top1_error_rate:.4f}%\n")
        error_file.write(f"Top-5 Error Rate: {top5_error_rate:.4f}%\n")
