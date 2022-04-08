import numpy as np
import os

import tvm
from tvm import relay, autotvm 
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
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
log_file = "%s-%s-B%d-%s-autotvm.json" % (model_name, layout, batch_size, target.kind.name)


# From PyTorch model to Torchscript
input_shape = [batch_size, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
scripted_model = torch.jit.trace(model, input_data).cuda()
scripted_model = scripted_model.eval()



# Set Tuning Options
#tuning_option = {"log_filename": log_file,
#                 "tuner": "xgb",
#                 "n_trial": 2000,
#                 "early_stopping": 600,
#                 "use_transfer_learning": False,
#                 "measure_option": autotvm.measure_option(
#                                   builder = autotvm.LocalBuilder(timeout = 10),
#                                   runner = autotvm.LocalRunner(number = 10, repeat = 1, min_repeat_ms = 1000)),}
tuning_option = {
    "log_filename": log_file,
    "tuner":"xgb",
    "n_trial": 100, #2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}


# Extract Search Tasks from the Network
print("Extract search tasks...")
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
# mod = convert_to_nhwc(mod)
ops = [relay.op.get("nn.batch_matmul"),
       relay.op.get("nn.dense"),
       relay.op.get("nn.conv2d")]

tasks = autotvm.task.extract_from_program(mod["main"], target = target, params = params, ops = ops)
for idx, task in enumerate(reversed(tasks)):
	print("=================Task %d======================" % (idx + 1))
	print(task)
	print("\n")





# Begin Tuning the Hyper-parameter
def tune_kernels(tasks, tuning_opt):
	

	measure_option = tuning_opt["measure_option"]
	tuner = tuning_opt["tuner"]
	print(tuner)
	n_trial = tuning_opt["n_trial"]
	early_stopping = tuning_opt["early_stopping"]
	log_filename = tuning_opt["log_filename"]
	use_transfer_learning = False
	tmp_log_file = log_filename + ".tmp"
	if os.path.exists(tmp_log_file):
		os.remove(tmp_log_file)
	
	for i, tsk in enumerate(reversed(tasks)):
		prefix = "[Task %2d/%2d])" % (i + 1, len(tasks))
       
        # Create Tuner
		if tuner == "xgb" or tuner == "xgb-rank":
			tuner_obj = XGBTuner(tsk, loss_type = "rank")
		if tuner == "ga":
			tuner_obj = GATuner(tsk, pop_size = 100)
		if tuner == "random":
			tuner_obj = RandomTuner(tsk)
		elif tuner == "gridsearch":
			tuner_obj = GridSearchTuner(tsk)

		if use_transfer_learning:
			if os.path.isfile(tmp_log_file):
				tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
		tsk_trial = min(n_trial, len(tsk.config_space))
		tuner_obj.tune(
			n_trial = tsk_trial,
			early_stopping = early_stopping,
			measure_option = measure_option,
			callbacks = [
				autotvm.callback.progress_bar(tsk_trial, prefix = prefix),
				autotvm.callback.log_to_file(tmp_log_file)
			]    
		)
		
		# pick best records 
		autotvm.record.pick_best(tmp_log_file, log_filename)
		os.remove(tmp_log_file)


#print("+++++++++++++++++++++")
#print(tuning_option["tuner"])
tune_kernels(tasks, tuning_option)



# Compile and Evaluate
print("Begin Compiling")
with autotvm.apply_history_best(log_file):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target = target, params = params)
       


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
