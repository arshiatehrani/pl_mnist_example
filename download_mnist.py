# download_mnist.py
from torchvision import datasets

# Download training set
datasets.MNIST(root='./data', train=True, download=True)

# Download test set  
datasets.MNIST(root='./data', train=False, download=True)

print("MNIST dataset downloaded successfully!")