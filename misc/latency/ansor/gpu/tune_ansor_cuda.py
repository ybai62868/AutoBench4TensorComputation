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
log_file = "%s-%s-B%d-%s-ansor.json" % (model_name, layout, batch_size, target.kind.name)


# From PyTorch model to Torchscript
input_shape = [batch_size, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
scripted_model = torch.jit.trace(model, input_data).cuda()
scripted_model = scripted_model.eval()



# Extract Search Tasks from the Network
print("Extract search tasks...")
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
mod = convert_to_nhwc(mod)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("=================Task %d (workload key: %s)======================" % (idx, task.workload_key))
    print(task.compute_dag)


# Begin Tuning the Hyper-parameter
print("Begin Tuning...")
measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(num_measure_trials = 18, runner = measure_ctx.runner, measure_callbacks=[auto_scheduler.RecordToFile(log_file)])
tuner.tune(tune_option)


# Compile and Evaluate
print("Begin Compiling...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level = 3, config = {"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target = target, params = params)



from tvm.contrib import graph_executor
dev = tvm.device(str(target), 0)
m = graph_executor.GraphModule(lib["default"](dev))

numpy_input = np.random.uniform(size=input_shape).astype(dtype)
tvm_input = tvm.nd.array(numpy_input, device=dev)
m.set_input(input_name, tvm_input)
m.run()
tvm_output = m.get_output(0)




print("Evaluate inference time cost...")
print(m.benchmark(dev, number=1, repeat=100))
