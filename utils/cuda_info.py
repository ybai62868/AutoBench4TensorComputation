import os
import subprocess
from subprocess import PIPE
from typing import List, Optional, Union


def query_gpu(names: Union[List[str], str]):
    if not isinstance(names, (list, tuple)):
        names = [names]
    result = subprocess.run(f'nvidia-smi -i 0 --query-gpu={",".join(names)} --format=csv,noheader,nounits'.split(),
                            stdin=PIPE, stdout=PIPE, check=True)
    results = [s.strip() for s in result.stdout.decode('utf-8').split(',')]
    if len(results) == 1:
        return results[0]
    else:
        return results



def query_device_name(short=False) -> str:
	ret = ""
	full_name = query_gpu("name")
	if short:
		short_name_dict = {
			"NVIDIA GeForce RTX 3060 Laptop GPU": "RTX3060L",
			"NVIDIA GeForce RTX 3090": "RTX390",
			"NVIDIA GeForce RTX 2080Ti": "RTX2080Ti",
			"Tesla T4": "T4",
		}
		ret = short_name_dict[full_name] if full_name in short_name_dict else full_name
	else:
		ret = full_name
	return ret

def device_query():
	pass

if __name__ == "__main__":
	print(query_device_name())
