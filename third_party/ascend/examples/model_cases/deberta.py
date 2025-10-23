import logging
import os

import torch
import torch_npu
import torch_npu._inductor

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

logging.basicConfig(level=logging.DEBUG)

torch.npu.config.allow_internal_format = False
torch.manual_seed(0)
torch.npu.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("./microsoft/deberta-v3-large")

sample_texts = ["This is a positive example.", "This might be negative."] * 128

model_ = AutoModelForTokenClassification.from_pretrained("./microsoft/deberta-v3-large", device_map="npu:0")
model_.eval()

inputs = tokenizer(
    sample_texts,
    max_length=512,
    padding="longest",
    truncation=True,
    return_tensors="pt",
    add_special_tokens=True
).to("npu:0")


def model(**model_inputs):
    with torch.no_grad():
        return model_(**model_inputs).logits

y = model(**inputs)
logging.info("result eager: " + str(torch.flatten(y)[:100]))

model_compiled = torch.compile(model_)

z = model_compiled(**inputs)
logging.info("result compiled: " + str(torch.flatten(z)[:100]))

torch.testing.assert_close(y, z, atol=1e-4, rtol=1e-4)
logging.info("deberta accuracy check pass!")