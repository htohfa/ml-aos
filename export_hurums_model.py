# Custom script to export the model to torchscript
# Mostly a copy of bin/export_model.py, but modified to work with Hurums file

from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from ml_aos.lightning import WaveNetSystem

# Name of the file
ckpt = "model_4to19_22to26_donutblur.ckpt"

# load the checkpoint and convert it to torchscript
script = WaveNetSystem.load_from_checkpoint(ckpt).to_torchscript()

# get the output file path
time_stamp = datetime.now(ZoneInfo("America/Los_Angeles"))
name = (
    "transfer_learning_unfrozen_"
    + str(time_stamp).split(".")[0].replace(" ", "_")
    + ".pt"
)

# print to terminal
print("Exporting", ckpt)
print("to", name)

# save the torchscript model
torch.jit.save(script, name)
