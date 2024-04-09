from transformers import AutoTokenizer

model_path = "/home/brian/Desktop/Mistral-7B-Instruct-v0.2/"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token #tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

texts = [
"<s> [INST] How does discord work? [/INST] Discord? It's like tuning into some subversive pirate radio. </s> [INST] So basically",
"</s> [INST] No not really."
]

#encoded = tokenizer.encode(text, padding=True)#tokenizer.encode("<s> [INST] How does [/INST] pirate radio.</s>")

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

#print(inputs)
#decoded = tokenizer.decode([28705])
#print(decoded)

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

chat_applied = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt")

print(chat_applied)