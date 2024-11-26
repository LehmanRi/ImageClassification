import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os


# Define the AlexNet class (ensure this matches the model used during training)
class AlexNet(nn.Module):
    def __init__(self, num_classes=8):  # Adjust number of classes as needed
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
            nn.Linear(256 * 7 * 7, 4096),
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

# Mapping class indices to WNID
class_to_wnid = {0: 'n01484850', 1: 'n01496331', 2: 'n01664065', 3: 'n01675722', 4: 'n01685808', 5: 'n01695060',
                 6: 'n01728920', 7: 'n01770081'}

if __name__ == '__main__':
    # Load the model and set to evaluation mode
    model = AlexNet(num_classes=8)
    # Load model weights, mapping to CPU if CUDA is not available
    model.load_state_dict(torch.load('best_alexnet_weights_8.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Directory containing test images
    test_dir = 'test_images'

    # Open files to save the results
    with open('test_error_rates.txt', 'w') as error_file, open('classification_results.txt', 'w') as class_file:
        error_file.write("Test Results:\n")
        class_file.write("Image Classification Results:\n")

        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        for image_name in os.listdir(test_dir):
            if image_name.endswith('.JPEG'):
                image_path = os.path.join(test_dir, image_name)

                # Extract true WNID and sequence number from filename
                wnid_label, sequence_number = image_name.split('_')

                # Open image and extract patches
                pil_image = Image.open(image_path).convert('RGB')
                patches = extract_patches(pil_image)

                # Store predictions for all patches
                avg_outputs = torch.zeros(1, 8).to(device)

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
                    torch.tensor([list(class_to_wnid.keys())[list(class_to_wnid.values()).index(wnid_label)]]).to(
                        device).view(1, -1).expand_as(pred))

                correct_top1 += correct[:1].reshape(-1).float().sum(0).item()
                correct_top5 += correct[:5].reshape(-1).float().sum(0).item()
                total_samples += 1

                # Get the top-1 predicted class and confidence
                top1_pred_class = pred[0, 0].item()
                predicted_wnid = class_to_wnid[top1_pred_class]
                top1_confidence = avg_outputs[0, top1_pred_class].item()

                # Print and save results for each image
                print(
                    f"Image: {image_name}, True WNID: {wnid_label}, Sequence: {sequence_number}, Predicted WNID: {predicted_wnid}, Confidence: {top1_confidence:.4f}")
                class_file.write(
                    f"{image_name}, True WNID: {wnid_label}, Sequence: {sequence_number}, Predicted WNID: {predicted_wnid}, Confidence: {top1_confidence:.4f}\n")

        # Calculate error rates
        top1_error_rate = 100 - (correct_top1 / total_samples) * 100
        top5_error_rate = 100 - (correct_top5 / total_samples) * 100
        print(f"Top-1 Error Rate: {top1_error_rate:.4f}%, Top-5 Error Rate: {top5_error_rate:.4f}%")
        error_file.write(f"Top-1 Error Rate: {top1_error_rate:.4f}%\n")
        error_file.write(f"Top-5 Error Rate: {top5_error_rate:.4f}%\n")
