"""
CNN Feature Learning with Step-by-Step Visualizations
Research Implementation for Computational Intelligence
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Create visualization directories
os.makedirs('visualizations/manual_steps', exist_ok=True)
os.makedirs('visualizations/filters', exist_ok=True)
os.makedirs('visualizations/feature_steps', exist_ok=True)

# ==================================================================
# 1. Enhanced Manual Convolution with Visualizations
# ==================================================================
def manual_convolution_demo():
    """Demonstrates CNN operations with step-by-step visualizations"""
    print("="*60 + "\nManual Convolution Visualization\n" + "="*60)
    
    # Create synthetic image
    image = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=np.float32)

    # Vertical edge detection kernel
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=np.float32)

    def plot_step(data, title, filename):
        """Helper function to plot and save steps"""
        plt.figure(figsize=(5,5))
        plt.imshow(data, cmap='gray', vmin=-3, vmax=3)
        plt.title(title)
        plt.colorbar()
        plt.savefig(f'visualizations/manual_steps/{filename}.png')
        plt.close()

    # Plot initial state
    plot_step(image, "Original Image", "01_original")
    plot_step(kernel, "Edge Detection Kernel", "02_kernel")

    # Convolution operation
    def conv2d(x, k):
        kh, kw = k.shape
        return np.array([
            [np.sum(x[i:i+kh, j:j+kw] * k) 
            for j in range(x.shape[1]-kw+1)]
            for i in range(x.shape[0]-kh+1)
        ])

    conv_result = conv2d(image, kernel)
    plot_step(conv_result, "Convolution Result", "03_convolution")

    # ReLU activation
    relu_result = np.maximum(0, conv_result)
    plot_step(relu_result, "After ReLU", "04_relu")

    # Translation equivariance
    shifted_image = np.roll(image, 1, axis=1)
    plot_step(shifted_image, "Shifted Input", "05_shifted_input")
    shifted_conv = conv2d(shifted_image, kernel)
    plot_step(shifted_conv, "Shifted Convolution", "06_shifted_conv")

    # Max pooling
    def maxpool2d(x, size=2):
        return np.array([
            [np.max(x[i:i+size, j:j+size])
            for j in range(0, x.shape[1], size)]
            for i in range(0, x.shape[0], size)
        ])

    pooled_result = maxpool2d(conv_result)
    plot_step(pooled_result, "Max Pooling Result", "07_pooling")

# ==================================================================
# 2. Enhanced CNN with Layer-wise Visualization
# ==================================================================
class VisualCNN(nn.Module):
    """CNN with built-in visualization hooks"""
    def __init__(self):
        super(VisualCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3),     # Conv1
            nn.ReLU(),                # ReLU1
            nn.MaxPool2d(2),          # Pool1
            nn.Conv2d(16, 32, 3),     # Conv2
            nn.ReLU(),                # ReLU2
            nn.MaxPool2d(2)           # Pool2
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.activations = {}

    def forward(self, x):
        # Layer 1
        x = self.feature_extractor[0](x)
        self.activations['conv1'] = x.detach()
        x = self.feature_extractor[1](x)
        self.activations['relu1'] = x.detach()
        x = self.feature_extractor[2](x)
        self.activations['pool1'] = x.detach()
        
        # Layer 2
        x = self.feature_extractor[3](x)
        self.activations['conv2'] = x.detach()
        x = self.feature_extractor[4](x)
        self.activations['relu2'] = x.detach()
        x = self.feature_extractor[5](x)
        self.activations['pool2'] = x.detach()
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def visualize_processing_steps(model, sample):
    """Visualizes all layer transformations for a sample image"""
    # Create directory for sample steps
    os.makedirs('visualizations/feature_steps/sample_processing', exist_ok=True)
    
    # Plot input
    plt.imshow(sample[0][0], cmap='gray')
    plt.title("Input Image")
    plt.savefig('visualizations/feature_steps/sample_processing/00_input.png')
    plt.close()

    # Forward pass with layer-wise capture
    with torch.no_grad():
        _ = model(sample)
    
    # Visualization parameters
    layer_steps = {
        'conv1': "Convolution Layer 1",
        'relu1': "ReLU Activation 1",
        'pool1': "Max Pooling 1",
        'conv2': "Convolution Layer 2",
        'relu2': "ReLU Activation 2",
        'pool2': "Max Pooling 2"
    }

    # Plot each processing step
    for i, (layer, title) in enumerate(layer_steps.items()):
        activations = model.activations[layer][0].cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.suptitle(title, fontsize=14)
        
        # Plot first 8 filters
        for j in range(min(8, activations.shape[0])):
            plt.subplot(2, 4, j+1)
            plt.imshow(activations[j], cmap='viridis')
            plt.title(f"Channel {j+1}")
            plt.axis('off')
        
        plt.savefig(f'visualizations/feature_steps/sample_processing/{i+1:02d}_{layer}.png')
        plt.close()

def train_and_visualize():
    """Training with comprehensive visualization"""
    # Dataset preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    
    # Initialize model
    model = VisualCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n" + "="*60 + "\nTraining Progress\n" + "="*60)
    for epoch in range(5):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Visualize learned features
    print("\n" + "="*60 + "\nFeature Analysis\n" + "="*60)
    
    # Visualize filters
    def plot_filters(weights, layer_name):
        filters = weights.detach().cpu().numpy()
        plt.figure(figsize=(12, 6))
        for i in range(min(16, filters.shape[0])):  # Corrected line
            plt.subplot(4, 4, i+1)
            plt.imshow(filters[i][0], cmap='gray')
            plt.axis('off')
        plt.savefig(f'visualizations/filters/{layer_name}_filters.png')
        plt.close()
    
    plot_filters(model.feature_extractor[0].weight, "layer1")
    plot_filters(model.feature_extractor[3].weight, "layer2")
    
    plot_filters(model.feature_extractor[0].weight, "layer1")
    plot_filters(model.feature_extractor[3].weight, "layer2")

    # Visualize processing steps
    sample = next(iter(torch.utils.data.DataLoader(train_set, batch_size=1)))[0]
    visualize_processing_steps(model, sample)

if __name__ == "__main__":
    manual_convolution_demo()
    train_and_visualize()