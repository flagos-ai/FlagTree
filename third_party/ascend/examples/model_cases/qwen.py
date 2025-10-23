import logging
import os

import torch
import torch_npu
import torch_npu._inductor

from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

logging.basicConfig(level=logging.DEBUG)

torch.npu.config.allow_internal_format = False
torch.manual_seed(0)
torch.npu.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct")

inputs = tokenizer("Hello, how to make China great again?", return_tensors="pt").to("npu:0")
model_ = AutoModelForCausalLM.from_pretrained("./Qwen2.5-0.5B-Instruct", device_map="npu:0")
model_.eval()


def model(**model_inputs):
    with torch.no_grad():
        return model_(**model_inputs).logits

y = model(**inputs)
logging.info("result eager: " + str(torch.flatten(y)[:100]))

model_compiled = torch.compile(model_)

z = model_compiled(**inputs)
logging.info("result compiled: " + str(torch.flatten(z)[:100]))

torch.testing.assert_close(y, z, atol=1e-4, rtol=1e-4)
logging.info("qwen accuracy check pass!")