# PyTorch Lightning MNIST example
MNIST example with PyTorch Lightning

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
- `--gpus -1`: Use all available GPUs (PyTorch Lightning will auto-detect)
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

#### 5. Quick Test Run (Minimal Resources)
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
| `--gpus` | Number of GPUs (0=CPU) | 1 | `0`, `1`, `-1` (all) |
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

### Viewing Training Metrics

After starting training, view metrics in TensorBoard:

```bash
# In a separate terminal, run:
tensorboard --logdir ./logs/lightning_logs

# Then open browser to:
# http://localhost:6006
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

### Out of Memory Errors
- Reduce `--batch_size` (try 16, 32, or 64)
- Reduce `--num_workers` (try 0, 2, or 4)
- Use `--model Linear` instead of `Conv` (smaller model)

### Slow Training
- Increase `--num_workers` for faster data loading (use `-1` to use all CPU cores)
- Use GPU if available (`--gpus 1`)
- Increase `--batch_size` if memory allows
- Reduce `--val_freq` to validate less frequently

### Data Download Issues
- Check internet connection
- Verify `--data_dir` path exists and is writable
- Manually download using `download_mnist.py`
