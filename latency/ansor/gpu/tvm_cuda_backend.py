import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from common import *
import torch
import torchvision
import logging

# logging.basicConfig(level=logging.INFO)


# Import Model from PyTorch Model Zoo
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=False).cuda()
model = model.eval()
batch_size = 1
layout = "NHWC"

dtype = "float32"
target = tvm.target.Target("cuda")
log_file = "%s-%s-B%d-%s.json" % (model_name, layout, batch_size, target.kind.name)


# From PyTorch model to Torchscript
input_shape = [batch_size, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
scripted_model = torch.jit.trace(model, input_data).cuda()
scripted_model = scripted_model.eval()

# Import the Graph to Relay
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
#mod = convert_to_nhwc(mod)




# Relay Build
dev = tvm.cuda(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)



# Execute the portable graph on TVM
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))
tvm_numpy = np.random.uniform(size=input_shape).astype(dtype)
tvm_input = tvm.nd.array(tvm_numpy, device=dev)
m.set_input(input_name, tvm_input)
m.run()
tvm_output = m.get_output(0)


print("Evaluate inference time cost...")
print(m.benchmark(dev, number=1, repeat=100))
