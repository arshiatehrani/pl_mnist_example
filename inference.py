"""
Inference script for MNIST digit classification.
Loads a trained model checkpoint and performs prediction on a single image.
"""

# torch: For tensor operations and PyTorch model loading/running
# numpy: For array manipulations, image data transformation (used with PIL Image)
# Conv: Model class for the convolutional MNIST model, defined in net.conv
# PIL.Image: To load and manipulate images
# matplotlib.pyplot: To visualize the images and predictions

import torch
import numpy as np
from net.conv import Conv
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model from checkpoint
# Note: Update the checkpoint path to match your trained model location
model = Conv.load_from_checkpoint("./models/state_dict/0615120754/epoch=4-step=30000.ckpt")

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()

# Load and preprocess the input image
# Note: Update './sample' to the path of your input image
# The (1 - ...) operation inverts the image (MNIST digits are white on black background)
image = 1 - np.array(Image.open('./sample'))

# Convert numpy array to PyTorch tensor
x = torch.Tensor(image)

# Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
# First unsqueeze adds batch dimension, second adds channel dimension (grayscale = 1 channel)
x = x.unsqueeze(0).unsqueeze(0)

# Perform inference without computing gradients (saves memory and speeds up inference)
with torch.no_grad():
    # Forward pass: get raw logits from the model
    y_hat = model(x)
    
    # Convert logits to probabilities using softmax
    # dim=1 indicates we're applying softmax across the class dimension (10 classes for MNIST)
    y_hat = torch.softmax(y_hat, dim=1)
    
    # Get the predicted class (digit 0-9) by finding the index with highest probability
    max_ind = int(torch.argmax(y_hat))

# Visualize the result
# Display the input image in grayscale
plt.imshow(image, cmap='gray')
# Show the prediction and confidence percentage in the title
plt.title(f'Guess is {max_ind} with {int(np.squeeze(y_hat.cpu().numpy())[max_ind]*100)}% confidence')
plt.show()