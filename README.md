# PyTorch Lightning MNIST example
MNIST example with PyTorch Lightning

## Features

- **PyTorch Lightning 2.0+ Compatible**: Updated to use the latest PyTorch Lightning API
- **Automatic GPU/CPU Detection**: Automatically falls back to CPU if GPU is not available
- **Flexible Resource Usage**: 
  - `--gpus -1`: Automatically use all available GPUs (or CPU if no GPU)
  - `--num_workers -1`: Automatically use all available CPU cores
- **TensorBoard Integration**: Real-time training metrics visualization with comprehensive logging
- **Multiple Model Architectures**: Support for both Linear and Convolutional models
- **Comprehensive Logging**: Automatic checkpointing, loss tracking, and accuracy metrics
- **Warning-Free Training**: Optimized to eliminate common warnings (pin_memory, persistent_workers)
- **Easy Data Management**: Automatic MNIST download with verification options
- **Production Ready**: Includes proper error handling, GPU fallback, and performance optimizations

## Virtual Environment Setup

### Initial Setup
```bash
# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TensorBoard (optional, for viewing training metrics)
pip install tensorboard
```

### GPU Support (Optional)
If you encounter warning messages that your current GPU is not compatible with the current PyTorch installation, uninstall the pytorch installations by:
```bash
pip uninstall torch torchvision torchaudio
```
and then refer to https://pytorch.org/ to get the required CUDA version suitable for your system.

## Running Training Locally

### Prerequisites
- Python 3.7+
- PyTorch and PyTorch Lightning installed
- (Optional) CUDA-enabled GPU for GPU training

### PyTorch Lightning 2.0+ Compatibility

This project is fully compatible with PyTorch Lightning 2.0+ and includes:
- **Updated API usage**: Uses `accelerator` and `devices` instead of deprecated `gpus` parameter
- **Modern hook names**: Uses `on_validation_epoch_end` and `on_test_epoch_end` instead of deprecated methods
- **Automatic GPU/CPU fallback**: When GPU is requested but not available, automatically falls back to CPU
- **Optimized DataLoader settings**: Automatic `pin_memory` and `persistent_workers` configuration based on device availability
- **TensorBoard graph logging**: Model architecture visualization with `example_input_array`

### Basic Command Structure
```bash
python train.py \
    --model <MODEL_NAME> \
    --dataloader MNIST \
    --data_dir <DATA_DIRECTORY> \
    --batch_size <BATCH_SIZE> \
    --epoch <NUM_EPOCHS> \
    --gpus <NUM_GPUS> \
    --logdir <LOG_DIRECTORY>
```

### Before You Start: Create Required Directories

**Important:** Create the `data` and `logs` directories before running training:

```bash
# Create directories for data and logs
# Windows PowerShell/CMD:
mkdir data
mkdir logs

# Linux/Mac:
mkdir -p data logs
```

**Why these directories?**
- `data/`: Stores the MNIST dataset (will be created automatically if downloading, but good to create beforehand)
- `logs/`: Stores training logs, checkpoints, and TensorBoard files (will be created automatically, but good to create beforehand)

**Note:** The training script will create subdirectories automatically (e.g., `logs/models/state_dict/`), but creating the base directories ensures everything is organized from the start.

### Training Configurations

#### 1. CPU-Only Training (No GPU)
Best for: Systems without GPU, testing, or small-scale experiments

```bash
# First, create required directories (if not already created)
mkdir -p data logs  # Linux/Mac
# or
mkdir data & mkdir logs  # Windows CMD

# Then run training
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 32 \
    --epoch 10 \
    --gpus 0 \
    --num_workers 4 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--gpus 0`: Use CPU only
- `--num_workers 4`: Use 4 CPU cores for data loading (adjust based on your CPU)
- `--batch_size 32`: Smaller batch size for CPU training

#### 2. Single GPU Training
Best for: Systems with one GPU, standard training setup

```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 64 \
    --epoch 20 \
    --gpus 1 \
    --num_workers 4 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--gpus 1`: Use 1 GPU
- `--batch_size 64`: Larger batch size for GPU (can increase to 128 or 256 if memory allows)
- `--num_workers 4`: Parallel data loading workers

#### 3. Multi-GPU Training (All Available GPUs)
Best for: Systems with multiple GPUs, faster training

```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 128 \
    --epoch 20 \
    --gpus -1 \
    --num_workers 8 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--gpus -1`: Use all available GPUs (PyTorch Lightning will auto-detect, or fall back to CPU if no GPU)
- `--batch_size 128`: Larger batch size (distributed across GPUs)
- `--num_workers 8`: More workers for parallel data loading

#### 4. Maximum CPU Utilization
Best for: CPU-only systems where you want to use all available cores

```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 32 \
    --epoch 10 \
    --gpus 0 \
    --num_workers -1 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--gpus 0`: CPU only
- `--num_workers -1`: Automatically use all available CPU cores for data loading
- Alternative: Set `--num_workers` to a specific number (e.g., `8`) to match your CPU core count

#### 5. Maximum CPU and GPU Utilization
Best for: High-performance systems with multiple GPUs, maximum resource utilization

```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 256 \
    --epoch 20 \
    --gpus -1 \
    --num_workers -1 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--gpus -1`: Use all available GPUs (PyTorch Lightning will auto-detect, or automatically fall back to CPU if no GPU is available)
- `--num_workers -1`: Automatically use all available CPU cores for data loading
- `--batch_size 256`: Large batch size (distributed across all GPUs)
- This configuration maximizes both GPU compute and CPU data loading parallelism

**Note on Automatic Fallback:**
- If you specify `--gpus -1` but no GPU is available, the system will automatically detect this and use CPU instead
- You'll see a message: `[info] No GPU available, falling back to CPU`
- This makes the same command work on both GPU and CPU systems

**Performance tips:**
- With multiple GPUs, batch size is effectively multiplied (256 batch size across 4 GPUs = 64 per GPU)
- All CPU cores handle data loading in parallel, keeping GPUs fed with data
- Ideal for workstations or servers with multiple GPUs and many CPU cores

#### 6. Quick Test Run (Minimal Resources)
Best for: Quick testing, debugging, or systems with limited resources

```bash
python train.py \
    --model Linear \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 16 \
    --epoch 2 \
    --gpus 0 \
    --num_workers 0 \
    --logdir ./logs \
    --lr 0.001
```

**Key parameters:**
- `--model Linear`: Simpler model (faster than Conv)
- `--batch_size 16`: Small batch size
- `--epoch 2`: Just 2 epochs for quick testing
- `--num_workers 0`: Single-threaded data loading

### Complete Parameter Reference

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--model` | Model architecture | Required | `Linear`, `Conv` |
| `--dataloader` | Dataset type | Required | `MNIST` |
| `--data_dir` | Dataset directory | Required | `./data` |
| `--batch_size` | Samples per batch | 1 | `32`, `64`, `128` |
| `--epoch` | Number of epochs | 20 | `10`, `20`, `50` |
| `--gpus` | Number of GPUs (0=CPU, -1=all) | 1 | `0` (CPU), `1` (1 GPU), `-1` (all GPUs, auto-fallback to CPU) |
| `--num_workers` | Data loading workers | 0 | `0`, `4`, `8`, `-1` (all CPU cores) |
| `--lr` | Learning rate | 0.001 | `0.0001`, `0.001`, `0.01` |
| `--val_ratio` | Validation split ratio | 0.2 | `0.1`, `0.2`, `0.3` |
| `--logdir` | Log directory | `./` | `./logs` |
| `--seed` | Random seed | 42 | Any integer |
| `--eval` | Run test evaluation | False | `True`, `False` |

### Recommended Configurations by System

#### Laptop/Desktop (CPU only)
```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 32 \
    --epoch 10 \
    --gpus 0 \
    --num_workers 4 \
    --logdir ./logs
```

#### Laptop/Desktop (Single GPU)
```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 64 \
    --epoch 20 \
    --gpus 1 \
    --num_workers 4 \
    --logdir ./logs
```

#### Workstation (Multi-GPU)
```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 128 \
    --epoch 20 \
    --gpus -1 \
    --num_workers 8 \
    --logdir ./logs
```

#### High-Performance Workstation/Server (Maximum Utilization)
```bash
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 256 \
    --epoch 20 \
    --gpus -1 \
    --num_workers -1 \
    --logdir ./logs
```

**Note:** This configuration uses all available GPUs and all available CPU cores for maximum performance.

### Data Download

The MNIST dataset will be automatically downloaded on the first run to the directory specified by `--data_dir`. 

#### Pre-download Data (Recommended)

To download the MNIST dataset beforehand and verify the data:

**Step 1: Run the download script**
```bash
# Make sure your virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac

# Run the download script
python download_mnist.py
```

**Step 2: Verify download**
After running the script, you should see:
```
MNIST dataset downloaded successfully!
```

**Step 3: Check data location**
The data will be downloaded to:
```
./data/
└── MNIST/
    ├── raw/
    │   ├── train-images-idx3-ubyte
    │   ├── train-labels-idx1-ubyte
    │   ├── t10k-images-idx3-ubyte
    │   └── t10k-labels-idx1-ubyte
    └── processed/
        ├── training.pt
        └── test.pt
```

**Step 4: Verify data files exist**
```bash
# Windows PowerShell
dir .\data\MNIST\raw\
dir .\data\MNIST\processed\

# Linux/Mac
ls -lh ./data/MNIST/raw/
ls -lh ./data/MNIST/processed/
```

**Step 5: (Optional) Inspect the data**
You can verify the dataset was downloaded correctly by checking:
- Training set: 60,000 images
- Test set: 10,000 images
- File sizes: Raw files should be ~9.9MB (images) and ~60KB (labels)

**Step 6: (Optional) Programmatically verify the data**
To check the data programmatically and see sample images:

```python
# verify_mnist.py
from torchvision import datasets
import matplotlib.pyplot as plt

# Load the datasets
train_data = datasets.MNIST(root='./data', train=True, download=False)
test_data = datasets.MNIST(root='./data', train=False, download=False)

# Print dataset information
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print(f"Image shape: {train_data[0][0].size}")
print(f"Number of classes: {len(train_data.classes)}")
print(f"Classes: {train_data.classes}")

# Display a sample image
image, label = train_data[0]
print(f"\nSample image label: {label}")
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.show()
```

Run the verification:
```bash
python verify_mnist.py
```

**Note:** If you want to download to a different directory, edit `download_mnist.py` and change the `root='./data'` parameter to your desired path, or modify the script to accept command-line arguments.

#### Automatic Download During Training

If you skip the pre-download step, the dataset will automatically download on the first training run. The download happens in `mnist.py` when the `MNISTDataModule` is initialized with `download=True`.

### Viewing Training Metrics with TensorBoard

TensorBoard provides real-time visualization of your training metrics, model architecture, and performance graphs.

#### Starting TensorBoard

**Option 1: View All Training Runs (Recommended)**

Open a **new terminal/PowerShell window** (keep training running in the original window) and run:

```powershell
# 1. Navigate to your project directory (IMPORTANT: Must be in project root, not logs folder)
# Replace <your-project-path> with your actual project directory path
cd <your-project-path>/pl_mnist_example
# Example on Windows: cd C:\Users\YourName\projects\pl_mnist_example
# Example on Linux/Mac: cd ~/projects/pl_mnist_example

# 2. Activate your virtual environment (REQUIRED)
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac
# or if using conda environment:
conda activate <your-env-name>

# 3. Start TensorBoard pointing to the lightning_logs directory
tensorboard --logdir ./logs/lightning_logs
```

**Important Notes:**
- You must be in the **project root directory** (where `train.py` is located), NOT inside the `logs` folder
- You must **activate your virtual environment** first (TensorBoard needs to be installed in the active environment)
- If you see "tensorboard: The term 'tensorboard' is not recognized", it means the virtual environment is not activated

**Option 2: View Specific Training Run**

If you want to view a specific run by timestamp:

```bash
tensorboard --logdir ./logs/lightning_logs/version_1218173008
```

(Replace `1218173008` with your actual timestamp from the training output)

#### Accessing TensorBoard

After starting TensorBoard, you should see output like:

```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

Then:

1. **Open your web browser**
2. **Navigate to**: `http://localhost:6006`
3. You should see the TensorBoard interface with your training metrics

#### What You'll See in TensorBoard

**Scalars Tab** (Main Metrics):
- **Training loss** - Training loss over time (logged every step)
- **Validation loss** - Validation loss (used for checkpointing, logged at validation intervals)
- **Validation Accuracy** - Model accuracy on validation set
- **Learning rate** - Current learning rate (if logged)
- **Epoch duration** - Time per epoch

**Graphs Tab**:
- **Model architecture** - Visual representation of your model (Conv or Linear)
- **Data flow** - How data moves through the network layers
- **Computation graph** - Full forward pass visualization

**Images Tab** (if you add image logging):
- Sample predictions and visualizations (if implemented)

#### TensorBoard Tips

**Real-Time Updates:**
- TensorBoard auto-refreshes every 30 seconds
- You can keep it open while training runs
- Metrics update automatically as training progresses

**Comparing Multiple Runs:**
- If you run training multiple times, TensorBoard shows all runs
- You can toggle runs on/off in the left sidebar to compare
- Useful for hyperparameter tuning and comparing different configurations

**Stopping TensorBoard:**
- Press `CTRL+C` in the terminal where TensorBoard is running

#### TensorBoard Command Options

```bash
# Basic usage (view all runs)
tensorboard --logdir ./logs/lightning_logs

# With custom port (if 6006 is busy)
tensorboard --logdir ./logs/lightning_logs --port 6007
# Then access: http://localhost:6007

# View specific run
tensorboard --logdir ./logs/lightning_logs/version_1218173008

# Load faster (disable fast loading for better compatibility)
tensorboard --logdir ./logs/lightning_logs --load_fast false

# Host on all interfaces (for remote access)
tensorboard --logdir ./logs/lightning_logs --host 0.0.0.0
```

#### Troubleshooting TensorBoard

**"tensorboard: The term 'tensorboard' is not recognized":**
- **Activate your virtual environment first**: `.\venv\Scripts\Activate.ps1` (Windows PowerShell) or `conda activate p` (if using conda)
- **Make sure you're in the project root directory**, not inside the logs folder
- **Verify TensorBoard is installed**: `pip install tensorboard` (run this in your activated virtual environment)
- **Check your current directory**: Run `pwd` (Linux/Mac) or `cd` (Windows) to see where you are. You should be in the project root where `train.py` is located

**"No dashboards are active":**
- Make sure training has run for at least one epoch
- Check that the log directory path is correct: `./logs/lightning_logs` (relative to project root)
- Verify logs are being written: `dir ./logs/lightning_logs` (Windows) or `ls ./logs/lightning_logs` (Linux/Mac)
- Make sure you're running TensorBoard from the project root directory

**Port 6006 already in use:**
- Use a different port: `tensorboard --logdir ./logs/lightning_logs --port 6007`
- Then access: `http://localhost:6007`

**"No scalar data":**
- Wait for the first validation step to complete
- Training needs to log at least one metric before TensorBoard can display it
- Check that training is actually running and logging metrics

**Quick Fix for "tensorboard not recognized":**
```powershell
# Step 1: Navigate to project root (replace with your actual project path)
cd <your-project-path>/pl_mnist_example
# Example: cd ~/projects/pl_mnist_example  (Linux/Mac)
# Example: cd C:\Users\YourName\projects\pl_mnist_example  (Windows)

# Step 2: Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac

# Step 3: Verify TensorBoard is installed (if not, install it)
pip install tensorboard

# Step 4: Run TensorBoard
tensorboard --logdir ./logs/lightning_logs
```

### Example: Complete Training Workflow

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac

# 2. Create directories
mkdir -p data logs

# 3. (Optional) Pre-download data
python download_mnist.py

# 4. Start training
python train.py \
    --model Conv \
    --dataloader MNIST \
    --data_dir ./data \
    --batch_size 32 \
    --epoch 10 \
    --gpus 0 \
    --num_workers 4 \
    --logdir ./logs \
    --eval True

# 5. (Optional) View results in TensorBoard
tensorboard --logdir ./logs/lightning_logs
```

## Running Training on Cloud (Cedana Compute)

*This section will be added with instructions for cloud deployment on Cedana Compute.*

## Troubleshooting

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version from https://pytorch.org/
- **Note**: If you use `--gpus -1` and no GPU is available, the system will automatically fall back to CPU

### PyTorch Lightning API Errors
- **"got an unexpected keyword argument 'gpus'"**: This project uses PyTorch Lightning 2.0+ API. Update your PyTorch Lightning: `pip install --upgrade pytorch-lightning`
- **"Support for validation_epoch_end has been removed"**: The code has been updated to use `on_validation_epoch_end`. Make sure you're using the latest version of the code.
- **"No supported gpu backend found!"**: The system will automatically fall back to CPU. This is expected behavior when no GPU is available.

### Out of Memory Errors
- Reduce `--batch_size` (try 16, 32, or 64)
- Reduce `--num_workers` (try 0, 2, or 4)
- Use `--model Linear` instead of `Conv` (smaller model)
- If using GPU, reduce batch size per GPU

### Slow Training
- Increase `--num_workers` for faster data loading (use `-1` to use all CPU cores)
- Use GPU if available (`--gpus 1` or `--gpus -1`)
- Increase `--batch_size` if memory allows
- Reduce `--val_freq` to validate less frequently
- Enable `persistent_workers` (automatically enabled when `num_workers > 0`)

### Warnings During Training
- **pin_memory warnings**: These are automatically handled - `pin_memory` is only used when GPU is available
- **persistent_workers warnings**: These are automatically enabled when `num_workers > 0` for better performance
- **TensorBoard graph warnings**: Model includes `example_input_array` for graph visualization

### TensorBoard Issues
- **TensorBoard not showing data**: Wait for at least one validation step to complete. Check that `./logs/lightning_logs` directory exists and contains event files.
- **Port already in use**: Use a different port with `--port 6007` flag.
- **"No dashboards are active"**: Ensure training has completed at least one epoch and validation step.

### Data Download Issues
- Check internet connection
- Verify `--data_dir` path exists and is writable
- Manually download using `download_mnist.py`
