import sys
import importlib
import os

if __name__ == "__main__":
    """
    Usage:
        ./scripts/run eval_hook.py ${eval_python_file} ${eval_config_file}
    """
    argv = sys.argv[1:]  # rm the script self name
    if len(argv) == 2:
        # for lang classifier
        run_path = str(argv[0]).replace("/", ".").replace(".py", "")
        cfg_path = str(argv[1]).replace("/", ".").replace(".py", "")

        run = importlib.import_module(run_path)
        cfg = importlib.import_module(cfg_path)

        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12311"
        # os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "1"


        # NCCL-related environment variables
        os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
        os.environ["FI_EFA_FORK_SAFE"] = "1"
        os.environ["FI_LOG_LEVEL"] = "1"
        os.environ["FI_PROVIDER"] = "efa"
        os.environ["FI_EFA_ENABLE_SHM_TRANSFER"] = "1"
        os.environ["NCCL_PROTO"] = "simple"

        run.main(cfg)
