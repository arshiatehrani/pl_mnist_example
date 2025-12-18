"""
Simple Fully-Connected Neural Network model for MNIST digit classification.
Uses two linear layers with ReLU activations.
"""

# torch.nn: PyTorch neural network modules (layers, activations, loss functions)
#          Used for building the model architecture (Linear layers)
import torch.nn as nn

# torch: Core PyTorch library for tensor operations and basic functions
#        Used for tensor operations, device management, and activation functions
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

class Linear(pl.LightningModule):
    """
    Simple Fully-Connected Neural Network for MNIST classification.
    
    Architecture:
    - Linear Layer 1: 784 -> 128 (with ReLU)
    - Linear Layer 2: 128 -> 10 (with ReLU)
    
    Input: (batch_size, 1, 28, 28) - grayscale MNIST images
    Output: (batch_size, 10) - logits for 10 digit classes
    
    Note: This is a simple baseline model. The Conv model typically performs better
          because it can learn spatial features through convolutions.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the linear model.
        
        Args:
            **kwargs: Dictionary containing hyperparameters:
                - lr: Learning rate for optimizer (float)
        """
        # Call parent class constructor to initialize LightningModule
        super().__init__()
        
        # self.lr: Float, learning rate for the optimizer
        #          Retrieved from kwargs, used in configure_optimizers()
        self.lr = kwargs.get('lr')

        # self.l1: First fully connected (linear) layer
        #          28 * 28 = 784: Input size (flattened MNIST image: 28x28 pixels)
        #          128: Output size (hidden layer with 128 neurons)
        #          Applies linear transformation: output = input * weight^T + bias
        self.l1 = nn.Linear(28 * 28, 128)
        
        # self.l2: Second fully connected (linear) layer
        #          128: Input size (from l1 output)
        #          10: Output size (one logit per digit class: 0-9)
        #          Final layer that produces class logits
        self.l2 = nn.Linear(128, 10)

        # save_hyperparameters: PyTorch Lightning method that saves all __init__ arguments
        #                      Allows checkpoint to restore model with same hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) - MNIST images
            
        Returns:
            x: Tensor of shape (batch_size, 10) - raw logits for each digit class
        """
        # Flatten input: Reshape from (batch_size, 1, 28, 28) to (batch_size, 784)
        # x.size(0): Batch size (first dimension)
        # -1: Automatically compute the flattened dimension (1 * 28 * 28 = 784)
        x = x.view(x.size(0), -1)
        
        # Apply first linear layer and ReLU activation
        # self.l1(x): Linear transformation: (batch, 784) -> (batch, 128)
        # torch.relu(...): ReLU activation: max(0, x) element-wise
        #                  Introduces non-linearity, allows model to learn complex patterns
        x = torch.relu(self.l1(x))
        
        # Apply second linear layer and ReLU activation
        # self.l2(x): Linear transformation: (batch, 128) -> (batch, 10)
        # torch.relu(...): ReLU activation (though typically final layer uses no activation or softmax)
        #                  Note: For classification, final layer often omits activation (raw logits)
        #                  Cross-entropy loss will apply softmax internally
        x = torch.relu(self.l2(x))
        
        return x


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