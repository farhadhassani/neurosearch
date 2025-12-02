#!/bin/bash
set -e

echo "🚀 Neurosearch GCP VM Setup Script"
echo "===================================="
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA drivers (if GPU VM)
echo "🎮 Installing NVIDIA GPU drivers..."
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Install CUDA (for A100)
echo "🔧 Installing CUDA toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.10
echo "🐍 Installing Python 3.10..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Install Python packages
echo "📚 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install sentence-transformers faiss-gpu transformers datasets pandas pyarrow scikit-learn tqdm rank_bm25 matplotlib seaborn plotly wandb pydantic

# Verify installations
echo ""
echo "✅ Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
if python3 -c "import torch; assert torch.cuda.is_available()"; then
    python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload neurosearch code: scp -r neurosearch/ user@vm-ip:~/"
echo "2. Run training: python3 train_on_vm.py"
