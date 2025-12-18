"""
Convolutional Neural Network model for MNIST digit classification.
Uses two convolutional blocks with max pooling, followed by a fully connected output layer.
"""

# torch.nn: PyTorch neural network modules (layers, activations, loss functions)
#          Used for building the model architecture (Conv2d, Linear, ReLU, MaxPool2d, Sequential)
import torch.nn as nn

# torch: Core PyTorch library for tensor operations and basic functions
#        Used for tensor operations, device management, and tensor creation
import torch

# pytorch_lightning: Framework for organizing PyTorch training code
#                   LightningModule base class provides training/validation/test structure
import pytorch_lightning as pl

# torch.nn.functional: Functional interface for neural network operations
#                     Used for loss functions (cross_entropy) and other operations
import torch.nn.functional as F

# numpy: Numerical computing library for array operations
#        Used for computing mean values of lists (accuracy, loss aggregation)
import numpy as np

class Conv(pl.LightningModule):
    """
    Convolutional Neural Network for MNIST classification.
    
    Architecture:
    - Conv Block 1: Conv2d(1->16) -> ReLU -> MaxPool2d
    - Conv Block 2: Conv2d(16->32) -> ReLU -> MaxPool2d
    - Output: Linear(32*7*7 -> 10)
    
    Input: (batch_size, 1, 28, 28) - grayscale MNIST images
    Output: (batch_size, 10) - logits for 10 digit classes
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the convolutional model.
        
        Args:
            **kwargs: Dictionary containing hyperparameters:
                - lr: Learning rate for optimizer (float)
        """
        # Call parent class constructor to initialize LightningModule
        super().__init__()
        
        # self.lr: Float, learning rate for the optimizer
        #          Retrieved from kwargs, used in configure_optimizers()
        self.lr = kwargs.get('lr')

        # self.conv1: First convolutional block
        #             Sequential container that applies layers in order
        #             Input: (batch, 1, 28, 28) -> Output: (batch, 16, 14, 14)
        self.conv1 = nn.Sequential(         
            # Conv2d: 2D convolutional layer
            #         in_channels=1: Input has 1 channel (grayscale)
            #         out_channels=16: Produces 16 feature maps
            #         kernel_size=5: 5x5 convolution kernel
            #         stride=1: Move kernel 1 pixel at a time
            #         padding=2: Add 2 pixels of zero-padding to maintain spatial size (28x28)
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            # ReLU: Rectified Linear Unit activation function
            #       Applies max(0, x) element-wise, introduces non-linearity
            nn.ReLU(),                      
            # MaxPool2d: Max pooling layer that downsamples by taking maximum in 2x2 regions
            #            kernel_size=2: 2x2 pooling window
            #            Reduces spatial dimensions by half: 28x28 -> 14x14
            nn.MaxPool2d(kernel_size=2),    
        )
        
        # self.conv2: Second convolutional block
        #             Sequential container for second set of conv operations
        #             Input: (batch, 16, 14, 14) -> Output: (batch, 32, 7, 7)
        self.conv2 = nn.Sequential(         
            # Conv2d: Second convolutional layer
            #         16: Input channels (from conv1 output)
            #         32: Output channels (more feature maps for higher-level features)
            #         5: Kernel size (5x5)
            #         1: Stride
            #         2: Padding (maintains 14x14 spatial size)
            nn.Conv2d(16, 32, 5, 1, 2),     
            # ReLU: Activation function for non-linearity
            nn.ReLU(),                      
            # MaxPool2d: Second pooling layer
            #            2: 2x2 pooling window
            #            Reduces spatial dimensions: 14x14 -> 7x7
            nn.MaxPool2d(2),                
        )

        # self.out: Final fully connected (linear) layer
        #          32 * 7 * 7 = 1568: Flattened feature size (32 channels * 7 height * 7 width)
        #          10: Output size (one logit per digit class: 0-9)
        self.out = nn.Linear(32 * 7 * 7, 10)

        # save_hyperparameters: PyTorch Lightning method that saves all __init__ arguments
        #                      Allows checkpoint to restore model with same hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) - MNIST images
            
        Returns:
            output: Tensor of shape (batch_size, 10) - raw logits for each digit class
        """
        # Apply first convolutional block: (batch, 1, 28, 28) -> (batch, 16, 14, 14)
        x = self.conv1(x)
        
        # Apply second convolutional block: (batch, 16, 14, 14) -> (batch, 32, 7, 7)
        x = self.conv2(x)
        
        # Flatten the output: Reshape from (batch, 32, 7, 7) to (batch, 32*7*7)
        # x.size(0): Batch size (first dimension)
        # -1: Automatically compute the flattened dimension (32 * 7 * 7 = 1568)
        x = x.view(x.size(0), -1)       
        
        # Apply final linear layer: (batch, 1568) -> (batch, 10)
        # Produces raw logits (unnormalized scores) for each of the 10 digit classes
        output = self.out(x)
        
        return output


    def training_step(self, batch, batch_idx):
        """
        Training step called for each batch during training.
        PyTorch Lightning automatically calls this method.
        
        Args:
            batch: Tuple of (images, labels) from training dataloader
            batch_idx: Integer index of current batch
            
        Returns:
            loss: Tensor, loss value for backpropagation
        """
        # Unpack batch into input images (x) and target labels (y)
        # x: Tensor of shape (batch_size, 1, 28, 28) - input images
        # y: Tensor of shape (batch_size,) - ground truth digit labels (0-9)
        x, y = batch
        
        # Forward pass: Get model predictions (logits)
        # y_hat: Tensor of shape (batch_size, 10) - raw logits for each class
        y_hat = self(x)
        
        # Compute cross-entropy loss between predictions and true labels
        # Cross-entropy is standard for multi-class classification
        # Automatically applies softmax and computes negative log-likelihood
        loss = F.cross_entropy(y_hat, y)
        
        # Log training loss to TensorBoard and console
        # loss.item(): Extract scalar value from tensor (required for logging)
        self.log('Training loss', loss.item())
        
        # Return loss for PyTorch Lightning to compute gradients and update weights
        return loss

    def on_validation_start(self):
        """
        Called at the start of each validation epoch.
        Initializes lists to accumulate metrics across all validation batches.
        """
        # self.losses: List to store loss values from each validation batch
        #             Will be averaged at end of validation epoch
        self.losses = []
        
        # self.accuracies: List to store accuracy values from each validation batch
        #                 Will be averaged at end of validation epoch
        self.accuracies = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step called for each batch during validation.
        Computes loss and accuracy for monitoring model performance.
        
        Args:
            batch: Tuple of (images, labels) from validation dataloader
            batch_idx: Integer index of current batch
            
        Returns:
            loss: Tensor, loss value (used for aggregation, not backpropagation)
        """
        # Unpack batch into input images and target labels
        x, y = batch
        
        # Forward pass: Get model predictions (logits)
        # Note: Variable named 'probs' but actually contains logits (not probabilities)
        probs = self(x)
        
        # Compute cross-entropy loss (same as training, but no gradients computed)
        loss = F.cross_entropy(probs, y)

        # Compute accuracy for this batch
        # acc: Tensor of shape (batch_size,) with 1.0 for correct, 0.0 for incorrect
        acc = self.accuracy(probs, y)
        
        # Convert accuracy tensor to list and add to accumulated accuracies
        # .cpu(): Move tensor to CPU (if on GPU)
        # .numpy(): Convert to numpy array
        # .tolist(): Convert to Python list
        # .extend(): Add all elements to the list (not append, since acc is a tensor)
        self.accuracies.extend(acc.cpu().numpy().tolist())
        
        # Add loss value to accumulated losses list
        # loss.item(): Extract scalar value from tensor
        self.losses.append(loss.item())
        
        return loss

    def validation_epoch_end(self, outputs):
        """
        Called at the end of each validation epoch.
        Computes and logs average metrics across all validation batches.
        
        Args:
            outputs: List of return values from validation_step (losses in this case)
        """
        # overall_acc: Float, mean accuracy across all validation batches
        #             Computes average of all individual accuracies collected
        overall_acc = np.mean(self.accuracies)
        
        # overall_loss: Float, mean loss across all validation batches
        #              Computes average of all batch losses
        overall_loss = np.mean(self.losses)
        
        # Log validation loss to TensorBoard and console
        # Used by ModelCheckpoint callback to determine "best" model
        self.log('Validation loss', overall_loss)
        
        # Log validation accuracy to TensorBoard and console
        # Provides human-readable performance metric
        self.log('Validation Accuracy', overall_acc)

    def on_test_start(self):
        """
        Called at the start of test evaluation.
        Initializes list to accumulate test accuracies.
        """
        # self.accuracies: List to store accuracy values from each test batch
        #                 Will be averaged at end of test epoch
        self.accuracies = []

    def test_step(self, batch, batch_idx):
        """
        Test step called for each batch during testing.
        Computes accuracy for final model evaluation.
        
        Args:
            batch: Tuple of (images, labels) from test dataloader
            batch_idx: Integer index of current batch
            
        Returns:
            acc: Tensor, accuracy values for this batch
        """
        # Unpack batch into input images and target labels
        x, y = batch
        
        # Forward pass: Get model predictions (logits)
        logits = self(x)
        
        # Compute accuracy for this batch
        # acc: Tensor of shape (batch_size,) with 1.0 for correct, 0.0 for incorrect
        acc = self.accuracy(logits, y)
        
        # Convert accuracy tensor to list and add to accumulated accuracies
        # .cpu().numpy().tolist(): Convert tensor to Python list
        # .extend(): Add all elements to the list
        self.accuracies.extend(acc.cpu().numpy().tolist())
        
        return acc

    def test_epoch_end(self, outputs):
        """
        Called at the end of test evaluation.
        Computes and logs average test accuracy.
        
        Args:
            outputs: List of return values from test_step (accuracies in this case)
        """
        # overall_acc: Float, mean accuracy across all test batches
        #             Final performance metric on held-out test set
        overall_acc = np.mean(self.accuracies)
        
        # Log test accuracy to TensorBoard and console
        # This is the final evaluation metric after training is complete
        self.log("Test Accuracy", overall_acc)

    def accuracy(self, logits, y):
        """
        Compute accuracy by comparing predicted class to true labels.
        
        Args:
            logits: Tensor of shape (batch_size, 10) - model output logits
            y: Tensor of shape (batch_size,) - true class labels (0-9)
            
        Returns:
            acc: Tensor of shape (batch_size,) - 1.0 for correct predictions, 0.0 for incorrect
        """
        # torch.argmax(logits, -1): Get predicted class for each sample
        #                          -1 means last dimension (class dimension)
        #                          Returns tensor of shape (batch_size,) with predicted class indices
        # torch.eq(..., y): Compare predictions to true labels element-wise
        #                   Returns boolean tensor: True where prediction matches label
        # .to(torch.float32): Convert boolean to float (True -> 1.0, False -> 0.0)
        # acc: Tensor of shape (batch_size,) with 1.0 for correct, 0.0 for incorrect
        acc = torch.eq(torch.argmax(logits, -1), y).to(torch.float32)
        
        return acc

    def configure_optimizers(self):
        """
        Configure optimizer for training.
        PyTorch Lightning automatically calls this to set up optimization.
        
        Returns:
            optimizer: PyTorch optimizer (Adam) configured with learning rate
        """
        # torch.optim.Adam: Adam optimizer (adaptive learning rate algorithm)
        # self.parameters(): All trainable parameters in the model (weights and biases)
        # lr=self.lr: Learning rate from initialization (typically 0.001)
        # Returns optimizer that will be used during training
        return torch.optim.Adam(self.parameters(), lr=self.lr)