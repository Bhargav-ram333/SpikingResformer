"""Probe LIF signal propagation across all 38 LIF nodes."""
import sys; sys.path.insert(0, '.')
from models.submodules import layers as _layers
from spikingjelly.activation_based import surrogate, neuron, functional
import torch
from timm.models import create_model
import models.spikingresformer

orig = _layers.LIF.__init__
def patched(self):
    neuron.LIFNode.__init__(self, tau=2., decay_input=True, v_threshold=1., v_reset=0.,
        surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
        backend='torch', store_v_seq=False)
_layers.LIF.__init__ = patched
model = create_model('spikingresformer_cifar', T=4, num_classes=10, img_size=32)
_layers.LIF.__init__ = orig
for m in model.modules():
    if hasattr(m, 'backend'):
        m.backend = 'torch'
model.eval()

hooks = []
layer_stats = {}

def make_hook(name):
    def hook_fn(module, input, output):
        x_in = input[0] if isinstance(input, tuple) else input
        layer_stats[name] = {
            'in_mean': x_in.float().mean().item(),
            'in_std':  x_in.float().std().item(),
            'out_mean':output.float().mean().item(),
            'out_std': output.float().std().item(),
            'in_zero': (x_in == 0).float().mean().item(),
        }
    return hook_fn

for name, m in model.named_modules():
    if isinstance(m, neuron.LIFNode):
        h = m.register_forward_hook(make_hook(name))
        hooks.append(h)

torch.manual_seed(42)
x = torch.randn(4, 3, 32, 32)
functional.reset_net(model)
with torch.no_grad():
    out = model(x)
functional.reset_net(model)
for h in hooks:
    h.remove()

names = list(layer_stats.keys())
print("LIF node signal propagation (all %d nodes):" % len(names))
print("%-55s  %8s  %8s  %7s  %8s" % ("name", "in_mean", "in_std", "in_zero", "out_mean"))
for n in names:
    s = layer_stats[n]
    print("%-55s  %8.4f  %8.4f  %7.3f  %8.4f" % (
        n, s['in_mean'], s['in_std'], s['in_zero'], s['out_mean']))
