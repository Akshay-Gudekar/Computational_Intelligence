import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.makedirs('visualizations/filters', exist_ok=True)
os.makedirs('visualizations/feature_maps', exist_ok=True)

# ==================================================================
# 1. Manual Convolution Demonstration
# ==================================================================
print("="*60 + "\nManual Convolution Demonstration\n" + "="*60)

# Define toy image and kernel
toy_image = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
], dtype=np.float32)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=np.float32)

def conv2d(image, kernel):
    """Manual 2D convolution implementation"""
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1
    output = np.zeros((o_h, o_w))
    for y in range(o_h):
        for x in range(o_w):
            output[y, x] = np.sum(image[y:y+k_h, x:x+k_w] * kernel)
    return output

# Perform convolution operations
conv_result = conv2d(toy_image, kernel)
relu_result = np.maximum(0, conv_result)
shifted_image = np.roll(toy_image, shift=1, axis=1)
shifted_conv = conv2d(shifted_image, kernel)

def maxpool2d(image, pool_size=2):
    """Manual max pooling implementation"""
    p_h, p_w = pool_size, pool_size
    i_h, i_w = image.shape
    o_h, o_w = i_h // p_h, i_w // p_w
    output = np.zeros((o_h, o_w))
    for y in range(o_h):
        for x in range(o_w):
            output[y, x] = np.max(image[y*p_h:(y+1)*p_h, x*p_w:(x+1)*p_w])
    return output

pool_result = maxpool2d(conv_result)

# Print results
print("Original Image:\n", toy_image)
print("\nConvolution Result:\n", conv_result)
print("\nAfter ReLU:\n", relu_result)
print("\nShifted Image Convolution:\n", shifted_conv)
print("\nMax Pooling Result:\n", pool_result)

# ==================================================================
# 2. CNN Model Implementation (MNIST Classification)
# ==================================================================
print("\n" + "="*60 + "\nCNN Model Implementation\n" + "="*60)

# Configuration
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Define CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # Input: 1x28x28, Output: 16x26x26
        self.pool1 = nn.MaxPool2d(2, 2)    # Output: 16x13x13
        self.conv2 = nn.Conv2d(16, 32, 3)  # Output: 32x11x11
        self.pool2 = nn.MaxPool2d(2, 2)    # Output: 32x5x5
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32*5*5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining Model...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# ==================================================================
# 3. Visualization of Filters and Feature Maps
# ==================================================================
print("\n" + "="*60 + "\nSaving Visualizations\n" + "="*60)

# Save first layer filters
filters = model.conv1.weight.data.cpu().numpy()
plt.figure(figsize=(10, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(filters[i][0], cmap='gray')
    plt.axis('off')
plt.suptitle('First Layer Filters')
plt.savefig('visualizations/filters/first_layer_filters.png')
plt.close()

# Save second layer filters
filters = model.conv2.weight.data.cpu().numpy()
plt.figure(figsize=(10, 8))
for i in range(32):
    plt.subplot(8, 4, i+1)
    plt.imshow(filters[i][0], cmap='gray')
    plt.axis('off')
plt.suptitle('Second Layer Filters')
plt.savefig('visualizations/filters/second_layer_filters.png')
plt.close()

# Hook for feature maps
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# Process and save feature maps
sample_image, _ = next(iter(train_loader))
sample_image = sample_image[0].unsqueeze(0)
output = model(sample_image)

# Save input image
plt.imshow(sample_image[0][0], cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.savefig('visualizations/feature_maps/input_image.png')
plt.close()

# Save first layer feature maps
conv1_maps = activations['conv1'][0].cpu().numpy()
plt.figure(figsize=(10, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(conv1_maps[i], cmap='gray')
    plt.axis('off')
plt.suptitle('First Layer Feature Maps')
plt.savefig('visualizations/feature_maps/first_layer_feature_maps.png')
plt.close()

# Save second layer feature maps
conv2_maps = activations['conv2'][0].cpu().numpy()
plt.figure(figsize=(10, 8))
for i in range(32):
    plt.subplot(8, 4, i+1)
    plt.imshow(conv2_maps[i], cmap='gray')
    plt.axis('off')
plt.suptitle('Second Layer Feature Maps')
plt.savefig('visualizations/feature_maps/second_layer_feature_maps.png')
plt.close()

print("Saved visualizations in:\n- visualizations/filters/\n- visualizations/feature_maps/")