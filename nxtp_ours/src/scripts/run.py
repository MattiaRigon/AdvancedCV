import os
import subprocess
import sys

# Inputs
input_script = sys.argv[1]
input_args = sys.argv[2:]

print(f"+ INPUT_ARGVS: {' '.join(input_args)}")

# Check if the input script exists
if not os.path.isfile(input_script):
    print(f"{input_script} not found")
    sys.exit(1)

# Environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12311"

# NCCL-related environment variables
os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
os.environ["FI_EFA_FORK_SAFE"] = "1"
os.environ["FI_LOG_LEVEL"] = "1"
os.environ["FI_PROVIDER"] = "efa"
os.environ["FI_EFA_ENABLE_SHM_TRANSFER"] = "1"
os.environ["NCCL_PROTO"] = "simple"

# Detect the number of GPUs
try:
    ngpus = int(subprocess.check_output(["nvidia-smi", "-L"]).decode().count('\n'))
except subprocess.CalledProcessError:
    print("Failed to detect GPUs or no GPU found.")
    sys.exit(1)

print(f"+ NGPUS: {ngpus}")
if ngpus == 0:
    print("No GPU found")
    sys.exit(1)

# Run torchrun with specified settings
torchrun_command = [
    "torchrun",
    f"--rdzv-endpoint=localhost:{os.environ['MASTER_PORT']}",
    "--nnode", "1",
    "--nproc_per_node", str(ngpus),
    input_script
] + input_args

print(f"+ Running command: {' '.join(torchrun_command)}")

try:
    subprocess.run(torchrun_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
