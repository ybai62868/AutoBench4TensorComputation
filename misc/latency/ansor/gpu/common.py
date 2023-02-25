import tvm
from tvm import relay, auto_scheduler

def convert_to_nhwc(mod):
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.conv3d": ["NDHWC", "default"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod


def dtype2torch(x):
    import torch
    return {"float32": torch.float32}[x]



