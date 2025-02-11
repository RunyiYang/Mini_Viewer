#!/bin/bash
#SBATCH --job-name=viewer
#SBATCH --nodelist=hala
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1          # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --mem=8G
#SBATCH --time=1:00:00         # Job timeout
#SBATCH --output=output_logs/env.log      # Redirect stdout to a log file
#SBATCH --error=output_logs/env.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email
# <<< mamba initialize <<<
eval "$(micromamba shell hook --shell bash)"
# export CC=/usr/bin/gcc-11.5
# export CXX=/usr/bin/g++-11.5
# export LD=/usr/bin/g++-11.5
export TORCH_CUDA_ARCH_LIST="8.6"

export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export CPLUS_INCLUDE_PATH=/opt/modules/nvidia-cuda-11.8.0/include

micromamba activate viewer
srun python run_viewer.py --ply models/point_cloud_30000.ply --port 8086