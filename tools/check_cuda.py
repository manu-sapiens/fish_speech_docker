import torch
import sys

print("Checking CUDA availability...")
print(f"CUDA is{'not ' if not torch.cuda.is_available() else ' '}available")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: CUDA is not available! The model will run on CPU which will be very slow.")
    sys.exit(1) if __name__ == "__main__" else None
