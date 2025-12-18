@echo off
REM Activate virtual environment and run training with maximum GPU and CPU utilization
call venv\Scripts\activate.bat
python train.py --model Conv --dataloader MNIST --data_dir ./data --batch_size 256 --epoch 20 --gpus -1 --num_workers -1 --logdir ./logs --lr 0.001
pause

