import torch
import logging
from typing import Union

logger = logging.getLogger(__name__)

def configure_gpu() -> None:
    """Configure GPU settings for PyTorch"""
    if torch.cuda.is_available():
        # Get GPU device count
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} GPU(s)")
        
        # Print device information for each GPU
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Enable cudnn autotuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set default device
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU found, using CPU instead")

def get_device() -> torch.device:
    """Get the current device (GPU if available, else CPU)
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def to_device(data: Union[torch.Tensor, list, tuple], device: torch.device) -> Union[torch.Tensor, list, tuple]:
    """Move data to specified device
    
    Args:
        data: Input data (tensor, list, or tuple)
        device: Target device
    
    Returns:
        Data moved to specified device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def main():
    configure_gpu()
    print(f"Current device: {get_device()}")

if __name__ == "__main__":
    main()
