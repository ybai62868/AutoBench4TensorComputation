import time
import torch
import torchvision
import transformers
import torch.nn as nn
from typing import Tuple, List, Dict

def model_info(name: str, batch_size: int=1, **kwargs) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
	if name == "resnet50":
		model.torchvision.models.resnet50(pretrained=True).eval().cuda()
		inputs = {
			"x" : torch.randn([batch_size, 3, 224, 224]).cuda()
		}
		return model, inputs
	elif name == "resnet18":
		model.torchvision.models.resnet18(pretrained=True).eval().cuda()
		inputs = {
			"x" : torch.randn([batch_size, 3, 224, 224]).cuda()
		}
		return model, inputs
	elif name == "resnet101":
		model.torchvision.models.resnet18(pretrained=True).eval().cuda()
		inputs = {
			"x" : torch.randn([batch_size, 3, 224, 224]).cuda()
		}
		return model, inputs
	elif name == "bert":
		config = transformers.GPT2Config()
		model = transformers.GPT2Model(config).eval().cuda()
		model.eval()
		vocab_size = 30522
		seq_length = kwargs.get("seq_length", 128)
		inputs = {
			"input_ids": torch.randint(0, vocab_size-1, size=[batch_size, seq_length]).cuda(),
			"attention_mask": torch.ones(size=[batch_size, seq_length], dtype=torch.int64).cuda(),
			"token_type_ids": torch.zeros(size=[batch_size, seq_length], dtype=torch.int64).cuda()
		}
		return model, inputs
	else:
		return ValueError("Can not recognize model: {}".format(name))


if __name__ == "__main__":
	for name in ["resnet18", "resnet50", "resnet101", "inception_v3", "mobilenet_v2", "bert", "gpt2"]:
		model, inputs = model_info(name)
		outputs = model(**inputs)
		repeats = 10
		torch.cuda.synchronize()
		t1 = time.time()
		for t in range(repeats):
			outputs = model(**inputs)
		torch.cuda.synchronize()
		t2 = time.time()
		print("{} {:.1f}".format(name, (t2-t1)/repeats*1000.0))
