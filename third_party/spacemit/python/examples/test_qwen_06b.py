from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

model_dir = "/mnt_ai_ws2/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True,
                                             torch_dtype=torch.float32, device_map="auto")

prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with flag_gems.use_gems():
    with torch.no_grad():
        gems_generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
        )
        gems_output_ids = gems_generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(gems_output_ids) - gems_output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        gems_thinking = tokenizer.decode(gems_output_ids[:index], skip_special_tokens=True).strip("\n")
        gems_content = tokenizer.decode(gems_output_ids[index:], skip_special_tokens=True).strip("\n")

        print(f"  thinking: {gems_thinking}")
        print(f"  content:  {gems_content}")
