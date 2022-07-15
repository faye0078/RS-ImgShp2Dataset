import torch
import torchvision
import os
from model.HRNet import get_seg_model
from model.make_fast_nas import fastNas
from config import obtain_retrain_args
# An instance of your model.

args = obtain_retrain_args()
# args.dataset = "GID-Vege5"

model = get_seg_model(args)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 5, 512, 512)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# output1 = model(torch.ones(1, 3, 224, 224))
# output2 = traced_script_module(torch.ones(1, 3, 224, 224))

if not os.path.exists("./export_model/" + args.dataset):
    os.makedirs("./export_model/" + args.dataset)
traced_script_module.save("./export_model/" + args.dataset + "/traced_hrnet_model.pt")