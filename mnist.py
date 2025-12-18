"""
MNIST Data Module for PyTorch Lightning
Handles loading, preprocessing, and splitting of the MNIST dataset into train/validation/test sets.
"""

# pytorch_lightning: Framework for organizing PyTorch code, provides LightningDataModule base class
#                   which standardizes data handling for training/validation/testing
import pytorch_lightning as pl

# torchvision.datasets: Contains pre-built dataset loaders (like MNIST)
# torchvision.transforms: Provides image transformation utilities (resize, normalize, augment, etc.)
from torchvision import datasets, transforms

# torch.utils.data.DataLoader: Efficiently loads data in batches with multiprocessing support
from torch.utils.data import DataLoader

# torch.utils.data.sampler.SubsetRandomSampler: Samples a subset of indices from a dataset
#                                              Used to split training data into train/validation sets
from torch.utils.data.sampler import SubsetRandomSampler

# numpy: Used for array operations, random number generation, and mathematical computations
#        Used here for shuffling indices and calculating split points
import numpy as np


class MNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for MNIST dataset.
    Manages dataset loading, preprocessing, and provides dataloaders for train/val/test splits.
    """

    def __init__(self, **kwargs):
        """
        Initialize the data module with configuration parameters.
        
        Args:
            **kwargs: Dictionary containing configuration parameters:
                - data_dir: Directory where MNIST dataset will be stored/downloaded
                - batch_size: Number of samples per batch
                - num_workers: Number of subprocesses for data loading (0 = main process only)
                - pin_memory: Whether to pin memory in DataLoader (faster GPU transfer)
                - val_ratio: Fraction of training data to use for validation (0.0 to 1.0)
        """
        # Call parent class constructor to initialize LightningDataModule
        super().__init__()

        # self.data_dir: String path to directory where MNIST dataset is stored or will be downloaded
        self.data_dir = kwargs.get('data_dir')
        
        # self.batch_size: Integer, number of images processed together in each batch
        #                 Larger batches use more memory but can be faster
        self.batch_size = kwargs.get('batch_size')
        
        # self.num_workers: Integer, number of parallel worker processes for loading data
        #                  Default is 0 (single-threaded). Higher values speed up data loading but use more CPU
        self.num_workers = kwargs.get('num_workers', 0)
        
        # self.pin_memory: Boolean, if True, data is pinned to GPU memory for faster transfer
        #                 Only effective when using GPU. Requires more RAM but speeds up training
        self.pin_memory = kwargs.get('pin_memory')
        
        # self.val_ratio: Float between 0.0 and 1.0, fraction of training data to use for validation
        #                 Example: 0.2 means 20% of training data becomes validation set
        self.val_ratio = kwargs.get('val_ratio')

        # error_msg: String message to display if validation ratio is out of valid range
        error_msg = "[!] valid_size should be in the range [0, 1]."
        
        # Assert that val_ratio is between 0 and 1, raise error with message if not
        assert ((self.val_ratio >= 0) and (self.val_ratio <= 1)), error_msg
        
        # self.do_transform: Compose object containing image transformations for training data
        #                    RandomAffine(15, (0.1, 0.1), (0.95, 1.05)) applies random rotation (±15°),
        #                    translation (±10%), and scaling (95-105%) for data augmentation
        #                    ToTensor() converts PIL Image to PyTorch tensor and scales to [0, 1]
        self.do_transform = transforms.Compose([transforms.RandomAffine(15, (0.1, 0.1), (0.95, 1.05)), transforms.ToTensor()])
        
        # self.no_transform: Compose object with minimal transformation (only ToTensor)
        #                    Used for validation and test sets (no augmentation needed)
        self.no_transform = transforms.Compose([transforms.ToTensor()])

        # self.dataset_train: MNIST dataset object for training
        #                     root: Directory to store dataset
        #                     train=True: Loads training split (60,000 images)
        #                     transform: Applies data augmentation transformations
        #                     download=True: Automatically downloads dataset if not present
        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, transform=self.do_transform, download=True)
        
        # self.dataset_val: MNIST dataset object for validation (uses same training split, will be subset)
        #                  train=True: Uses training split (we'll split it manually)
        #                  transform: No augmentation, just converts to tensor
        #                  download=True: Downloads if needed
        self.dataset_val = datasets.MNIST(root=self.data_dir, train=True, transform=self.no_transform, download=True)

        # num_train: Integer, total number of samples in the training dataset (typically 60,000 for MNIST)
        num_train = len(self.dataset_train)
        
        # indices: List of integers [0, 1, 2, ..., num_train-1], all possible indices for training data
        indices = list(range(num_train))
        
        # split: Integer, index where to split the dataset into train and validation
        #        Calculated as floor of (validation_ratio * total_samples)
        #        Example: if val_ratio=0.2 and num_train=60000, split=12000
        split = int(np.floor(self.val_ratio * num_train))

        # Shuffle the indices randomly to ensure train/val split is random (not sequential)
        # This ensures both sets have diverse samples, not just first/last N samples
        np.random.shuffle(indices)

        # train_idx: List of indices for training set (everything after the split point)
        #            Example: if split=12000, train_idx = indices[12000:] (48,000 samples)
        # valid_idx: List of indices for validation set (everything before the split point)
        #            Example: if split=12000, valid_idx = indices[:12000] (12,000 samples)
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # self.dataset_tr_indices: SubsetRandomSampler that samples only training indices
        #                          Used by DataLoader to only access training subset
        self.dataset_tr_indices = SubsetRandomSampler(train_idx)
        
        # self.dataset_val_indices: SubsetRandomSampler that samples only validation indices
        #                           Used by DataLoader to only access validation subset
        self.dataset_val_indices = SubsetRandomSampler(valid_idx)

        # self.dataset_test: MNIST dataset object for testing
        #                    train=False: Loads official test split (10,000 images)
        #                    transform: No augmentation, just converts to tensor
        #                    download=True: Downloads if needed
        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, transform=self.no_transform, download=True)


    def train_dataloader(self):
        """
        Returns DataLoader for training set.
        Called automatically by PyTorch Lightning during training.
        
        Returns:
            DataLoader: Iterator that yields batches of (image, label) tuples for training
        """
        # Return DataLoader configured for training:
        # - dataset_train: Full training dataset (60,000 samples)
        # - batch_size: Number of samples per batch
        # - sampler: Only samples indices from training subset (excludes validation indices)
        # - num_workers: Number of parallel data loading processes
        # - pin_memory: Pins data to GPU memory for faster transfer (if using GPU)
        return DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=self.dataset_tr_indices, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        """
        Returns DataLoader for validation set.
        Called automatically by PyTorch Lightning during validation.
        
        Returns:
            DataLoader: Iterator that yields batches of (image, label) tuples for validation
        """
        # Return DataLoader configured for validation:
        # - dataset_val: Full training dataset (60,000 samples, but sampler limits access)
        # - batch_size: Number of samples per batch
        # - sampler: Only samples indices from validation subset (excludes training indices)
        # - num_workers: Number of parallel data loading processes
        # - pin_memory: Pins data to GPU memory for faster transfer (if using GPU)
        return DataLoader(self.dataset_val, batch_size=self.batch_size, sampler=self.dataset_val_indices, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        """
        Returns DataLoader for test set.
        Called automatically by PyTorch Lightning during testing.
        
        Returns:
            DataLoader: Iterator that yields batches of (image, label) tuples for testing
        """
        # Return DataLoader configured for testing:
        # - dataset_test: Official test dataset (10,000 samples, separate from training)
        # - batch_size: Number of samples per batch
        # - num_workers: Number of parallel data loading processes
        # - shuffle=False: Don't shuffle test data (deterministic order for reproducibility)
        # - pin_memory: Pins data to GPU memory for faster transfer (if using GPU)
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
