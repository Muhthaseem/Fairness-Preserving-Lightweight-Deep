import torch
import sys

def check_gpu_status():
    print("\n" + "="*60)
    print("GPU ACCELERATION STATUS")
    print("="*60)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ CUDA Available: YES")
        print(f"   Device Count: {device_count}")
        print(f"   Current Device: {current_device} ({device_name})")
        
        # Memory Info
        total_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(current_device) / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(current_device) / (1024**3)
        
        print(f"   Total VRAM:     {total_mem:.2f} GB")
        print(f"   Allocated:      {allocated_mem:.2f} GB")
        print(f"   Reserved:       {reserved_mem:.2f} GB")
        
        # Compute Capability
        cap = torch.cuda.get_device_capability(current_device)
        print(f"   Compute Cap:    {cap[0]}.{cap[1]}")
        
        # Test Tensor
        try:
            x = torch.ones(1).cuda()
            print(f"   Tensor Test:    PASSED")
        except Exception as e:
            print(f"   Tensor Test:    FAILED ({e})")
            
        print(f"\n   >> READY FOR TRAINING ON {device_name} <<")
    else:
        print("❌ CUDA Available: NO")
        print("   Running on CPU. Training will be extremely slow.")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    check_gpu_status()
