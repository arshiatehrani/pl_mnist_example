"""
Training script for MNIST digit classification using PyTorch Lightning.
Supports multiple model architectures and provides comprehensive training configuration.
"""

# argparse: Parses command-line arguments to configure training hyperparameters
import argparse

# datetime: Generates timestamp strings for organizing saved models and logs
import datetime

# os: Handles file system operations (creating directories, joining paths)
import os

# random: Used for random number generation (though seeding is handled by PyTorch Lightning)
import random

# time: Used for timing operations (though not actively used in current code)
import time


# pytorch_lightning: Main framework for organizing PyTorch training code
#                   Provides Trainer class that handles training loop, validation, logging, etc.
import pytorch_lightning as pl

# ModelCheckpoint: Callback that automatically saves model checkpoints during training
#                  Can save best models based on validation metrics
from pytorch_lightning.callbacks import ModelCheckpoint

# SimpleProfiler: Profiles training to identify bottlenecks and measure execution time
from pytorch_lightning.profilers import SimpleProfiler

# TensorBoardLogger: Logs training metrics, losses, and graphs to TensorBoard for visualization
from pytorch_lightning.loggers import TensorBoardLogger

# MNISTDataModule: Custom data module that handles MNIST dataset loading and preprocessing
from mnist import MNISTDataModule

# Linear: Simple fully-connected neural network model (784 -> 128 -> 10)
from net.linear import Linear

# Conv: Convolutional neural network model with conv layers and pooling
from net.conv import Conv


# MODEL_DIRECTORY: Dictionary mapping model name strings to model class constructors
#                 Allows selecting model architecture via command-line argument
#                 Keys: "Linear" or "Conv" (user-specified)
#                 Values: Model class (Linear or Conv)
MODEL_DIRECTORY = {
    "Linear": Linear,
    "Conv": Conv
}


# DATALOADER_DIRECTORY: Dictionary mapping dataloader name strings to data module classes
#                       Allows selecting dataset via command-line argument
#                       Keys: Dataset name (e.g., "MNIST")
#                       Values: Data module class (e.g., MNISTDataModule)
DATALOADER_DIRECTORY = {
    'MNIST': MNISTDataModule,
} 

if __name__ == "__main__":
    """
    Main training script execution.
    Parses arguments, initializes model and data, configures trainer, and runs training.
    """
    
    # parser: ArgumentParser object that defines and parses command-line arguments
    # RawTextHelpFormatter allows multi-line help text for better documentation
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    # --model: String, name of model architecture to train ("Linear" or "Conv")
    #          Required argument, must match a key in MODEL_DIRECTORY
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    
    # --eval: Boolean, whether to run test evaluation after training completes
    #         If True, evaluates on test set using best checkpoint
    parser.add_argument('--eval', help='Whether to test model on the best iteration after training'
                        , default=False, type=bool)
    
    # --dataloader: String, name of dataset/data module to use ("MNIST")
    #               Required argument, must match a key in DATALOADER_DIRECTORY
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    
    # --load: String path, directory of pre-trained model checkpoint to load
    #         If None, training starts from randomly initialized weights
    #         Loads only model weights (not optimizer state, epoch number, etc.)
    parser.add_argument("--load",
                        help="Directory of pre-trained model weights only,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    
    # --resume_from_checkpoint: String path, directory of full checkpoint to resume from
    #                           If None, training starts fresh
    #                           Resumes training state (weights, optimizer, epoch, etc.)
    parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained checkpoint including hyperparams,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    
    # --data_dir: String path, directory where dataset is stored or will be downloaded
    #             Required argument, MNIST will download here if not present
    parser.add_argument('--data_dir', help='Directory of your Dataset', required=True, default=None)
    
    # --gpus: Integer, number of GPUs to use for training (0 = CPU only, 1+ = GPU training)
    #         Default is 1 GPU
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=1, type=int)
    
    # --batch_size: Integer, number of samples processed together in each batch
    #               Default is 1 (very small, typically use 32, 64, or 128)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    
    # --epoch: Integer, number of complete passes through the training dataset
    #          Default is 20 epochs
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    
    # --num_workers: Integer, number of parallel worker processes for data loading
    #                Default is 0 (single-threaded), higher values speed up data loading
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    
    # --val_freq: Float, how often to run validation within a training epoch
    #            0.1 = validate 10% through each epoch (10 times per epoch)
    #            0.25 = validate 4 times per epoch, 1.0 = validate once at end of epoch
    parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=0.1, type=float)
    
    # --logdir: String path, base directory for saving logs, checkpoints, and TensorBoard files
    #           Default is current directory ("./")
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    
    # --lr: Float, learning rate for optimizer (controls step size during gradient descent)
    #       Default is 0.001, typical range is 0.0001 to 0.01
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    
    # --display_freq: Integer, frequency (in batches) to log images to TensorBoard
    #                 Default is every 64 batches (not actively used in current models)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    
    # --seed: Integer, random seed for reproducibility (sets random state for Python, NumPy, PyTorch)
    #         Default is 42, same seed produces same results
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    
    # --clip_grad_norm: Float, maximum gradient norm for gradient clipping (prevents exploding gradients)
    #                  0.0 means no clipping, typical values are 1.0, 5.0, or 10.0
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    
    # --pin_memory: Boolean, whether to pin data in memory for faster GPU transfer
    #              Default is True, speeds up training when using GPU
    parser.add_argument('--pin_memory', help='Whether to utilize pin_memory in dataloader', type=bool, default=True)
    
    # --val_ratio: Float between 0.0 and 1.0, fraction of training data to use for validation
    #              Default is 0.2 (20% of training data becomes validation set)
    parser.add_argument('--val_ratio', help='Float between [0, 1] to indicate the percentage of train dataset to validate on', type=float, default=0.2)


    # args: Namespace object containing all parsed command-line arguments
    args = parser.parse_args()
    
    # dict_args: Dictionary version of args for easier access and passing to functions
    #            Converts Namespace to dict: {'model': 'Conv', 'batch_size': 32, ...}
    dict_args = vars(args)
    
    # Set random seed for all random number generators (Python, NumPy, PyTorch, CUDA)
    # Ensures reproducible results across runs with same seed
    pl.seed_everything(dict_args['seed'])
    
    # Print which model architecture will be trained
    print(f"[info] Model: {dict_args['model']}")
    
    # Verify that the specified model name exists in MODEL_DIRECTORY
    # Raises AssertionError if model name is invalid
    assert dict_args['model'] in MODEL_DIRECTORY
    
    # Initialize model: either load from checkpoint or create new instance
    if dict_args['load'] is not None:
        # Load pre-trained model from checkpoint file
        # **dict_args passes all arguments (lr, etc.) to model constructor
        model = MODEL_DIRECTORY[dict_args['model']].load_from_checkpoint(dict_args['load'], **dict_args)
    else:
        # Create new model instance with random initialization
        # **dict_args unpacks dictionary as keyword arguments to model constructor
        model = MODEL_DIRECTORY[dict_args['model']](**dict_args)

    # Generate timestamp string for organizing saved models (format: MMDDHHMMSS)
    # Example: "1225143055" = December 25, 2:30:55 PM
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    
    # weight_save_dir: String path where model checkpoints will be saved
    #                 Structure: <logdir>/models/state_dict/<timestamp>/
    #                 Example: "./models/state_dict/1225143055/"
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))

    # Create the checkpoint directory if it doesn't exist
    # exist_ok=True prevents error if directory already exists
    os.makedirs(weight_save_dir, exist_ok=True)
    
    # Print where checkpoints will be saved
    print(f"[info] Saving weights to : {weight_save_dir}")

    # -----------------------------------------------------------------------------------------------------
    # -------------------------------------- Checkpoint Callback ------------------------------------------
    # checkpoint_callback: ModelCheckpoint callback that automatically saves model checkpoints
    #                     dirpath: Where to save checkpoints
    #                     save_top_k: Keep only the top 5 best models (based on validation loss)
    #                     verbose: Print messages when saving checkpoints
    #                     monitor: Metric to monitor for "best" model ("Validation loss")
    #                     mode: "min" means lower validation loss is better
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Validation loss", mode="min"
    )
    # -------------------------------------- Checkpoint Callback ------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    # Verify that the specified dataloader name exists in DATALOADER_DIRECTORY
    # Raises AssertionError if dataloader name is invalid
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    
    # data_module: Data module instance that handles dataset loading and preprocessing
    #              **dict_args passes configuration (data_dir, batch_size, num_workers, etc.)
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)
    
    # Print which dataloader is being used
    print(f"[info] Using dataloader: {dict_args['dataloader']}")

    # profiler: SimpleProfiler instance that measures execution time of training steps
    #          Helps identify bottlenecks in training pipeline
    profiler = SimpleProfiler()
    
    # logger: TensorBoardLogger instance that logs metrics to TensorBoard
    #         save_dir: Base directory for logs
    #         version: Unique identifier (timestamp) for this training run
    #         name: Subdirectory name ("lightning_logs")
    #         log_graph: Whether to log model computation graph
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    
    # trainer: PyTorch Lightning Trainer that orchestrates the entire training process
    #          callbacks: List of callbacks (checkpoint saving, etc.)
    #          val_check_interval: How often to validate (0.1 = 10% through each epoch)
    #          deterministic: Ensures reproducible results (slower but consistent)
    #          gpus: Number of GPUs to use
    #          profiler: Performance profiler for timing analysis
    #          logger: TensorBoard logger for metrics visualization
    #          max_epochs: Maximum number of training epochs
    #          log_every_n_steps: Log metrics every N training steps
    #          gradient_clip_val: Maximum gradient norm (0 = no clipping)
    #          resume_from_checkpoint: Path to checkpoint to resume from (None = start fresh)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        val_check_interval=dict_args['val_freq'],
        deterministic=True,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        resume_from_checkpoint=dict_args['resume_from_checkpoint']
    )

    # Print training start message
    print(f"[info] Starting training")
    
    # trainer.fit: Main training function that runs the training loop
    #              Trains model on training data, validates on validation data
    #              Automatically handles epochs, batching, optimization, logging, checkpointing
    trainer.fit(model, data_module)

    # Optionally evaluate model on test set after training completes
    if dict_args['eval']:
        # trainer.test: Evaluates model on test dataset
        #               ckpt_path='best': Uses the best checkpoint (lowest validation loss)
        #               datamodule: Provides test dataloader
        trainer.test(model, ckpt_path='best', datamodule=data_module)
    else:
        # Skip test evaluation if --eval flag is False
        print("Evaluation skipped")
