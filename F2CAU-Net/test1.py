import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the fuzzy U-Net model
from FuzzyConv2 import FuzzyUNet  # Replace "your_module_or_file" with the correct import path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = FuzzyUNet().to(device)

# Load pre-trained weights (if available)
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
# model.eval()

# Function to create a random test image
def create_random_test_image(size):
    # Generate random image data with float data type
    return np.random.rand(size, size).astype(np.float32)  # Convert data type to float32


# Function to preprocess the image and convert it to a PyTorch tensor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Generate a random test image with float data type
image_size = 256
test_image = create_random_test_image(image_size)

# Preprocess the test image
test_image_tensor = preprocess_image(test_image)

# Perform image segmentation using the fuzzy U-Net model
with torch.no_grad():
    model.eval()
    output = model(test_image_tensor)

# Convert the output tensor to a numpy array and reshape to (256, 256)
output_array = output.cpu().squeeze(0).numpy()
segmentation_mask = np.argmax(output_array, axis=0)

# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(segmentation_mask, cmap="jet")
plt.axis("off")

plt.show()
