"""6L Best (AUC=0.641) Viser 시각화"""
import os, sys, glob, torch
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
VGGT_ROOT = str(HERE / 'vggt-fresh' / 'vggt-main')
sys.path.insert(0, os.path.join(VGGT_ROOT, 'training'))
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, str(HERE))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from fla.layers import GatedLinearAttention
import torch.nn as nn

device = "cuda"; dtype = torch.bfloat16

class FLABlock(nn.Module):
    def __init__(self, ob, m):
        super().__init__()
        self.mixer=m;self.norm1=ob.norm1;self.ls1=ob.ls1;self.norm2=ob.norm2;self.mlp=ob.mlp;self.ls2=ob.ls2
    def forward(self, x, pos=None):
        o=self.mixer(self.norm1(x))
        if isinstance(o,tuple):o=o[0]
        x=x+self.ls1(o);x=x+self.ls2(self.mlp(self.norm2(x)));return x

print("Loading 6L (AUC=0.641)...")
pretrained = torch.load('/root/.cache/torch/hub/checkpoints/model.pt', map_location='cpu', weights_only=False)
model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
    enable_camera=True, enable_depth=True, enable_point=True, enable_track=False).to(device)
model.load_state_dict(pretrained, strict=False)

for idx in [2, 5, 7, 9, 18, 20]:
    gla = GatedLinearAttention(hidden_size=1024, num_heads=16,
        use_output_gate=True, use_short_conv=False, mode='chunk').to(device)
    wrapped = FLABlock(model.aggregator.global_blocks[idx], gla).to(device)
    model.aggregator.global_blocks[idx] = wrapped

ckpt = torch.load(str(HERE / 'weights' / '6L_best_AUC0641.pt'), map_location='cpu', weights_only=False)
model_sd = model.state_dict()
for k, v in ckpt['model'].items():
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
model.load_state_dict(model_sd, strict=False)
model.eval()
print("Model loaded.")

image_names = sorted(glob.glob(os.path.join(VGGT_ROOT, 'examples/kitchen/images/*.png')))
print(f"{len(image_names)} images")
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

print("Starting Viser on port 8080...")
from demo_viser import viser_wrapper
viser_wrapper(predictions, port=8080, init_conf_threshold=25.0, use_point_map=True)
