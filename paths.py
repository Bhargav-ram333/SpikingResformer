"""
paths.py -- where the frozen backbone checkpoint and the CUB-200-2011 dataset live.

Every script used to hard-code the same two Windows paths. They are now read here, from environment
variables when set, falling back to the original locations, so existing runs behave exactly as before:

    SRF_CKPT_PATH   SpikingResformer-Ti checkpoint (spikingresformer_ti.pth)
    CUB_DIR         CUB_200_2011 folder (expects processed_attributes.csv and images/)

Example (bash):        export CUB_DIR=/data/CUB_200_2011
Example (PowerShell):  $env:CUB_DIR = "D:\data\CUB_200_2011"
"""
import os

_DEFAULT_CKPT = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
_DEFAULT_CUB = r"C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011"

SRF_CKPT_PATH = os.environ.get("SRF_CKPT_PATH", _DEFAULT_CKPT)
CUB_DIR = os.environ.get("CUB_DIR", _DEFAULT_CUB)
