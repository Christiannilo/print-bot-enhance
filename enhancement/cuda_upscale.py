"""Upscale Real-ESRGAN x4plus por CUDA (GTX 960 sm_52, torch 2.4.1+cu121).
Uso: cuda_upscale.py SRC DEST OUTSCALE"""
import sys, time
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np

t0 = time.time()
rr = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
up = RealESRGANer(scale=4, model_path='/opt/realesrgan-models/realesrgan-x4plus/RealESRGAN_x4plus.pth',
                  model=rr, tile=128, device='cuda')
img = np.array(Image.open(sys.argv[1]).convert('RGB'))
out = up.enhance(img, outscale=float(sys.argv[3]))[0]
Image.fromarray(out.astype('uint8')).save(sys.argv[2])
print(f"OK {time.time()-t0:.1f}s vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
