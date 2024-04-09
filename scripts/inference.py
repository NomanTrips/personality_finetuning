from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "/home/brian/Desktop/Llama-2-7b-chat-hf/" #"meta-llama/Llama-2-7b-chat-hf" "./ckpt/silverhand_dpo_3_16_816/"
prompt = "<s>[INST] You are a helpful assistant. I have a stuck zipper on my jacket, what can I do to get it unstuck? [/INST]"
import time

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

start_time = time.time()
output = model.generate(**model_inputs, max_length=256)
end_time = time.time()
total_time = end_time - start_time
print(f"Total inference time: {total_time} seconds")

print(tokenizer.decode(output[0], skip_special_tokens=True))
