""" Imports used to generate responses """
import torch
import torch._dynamo # Importato per ignorare gli errori di compilazione
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.suppress_errors = True # Impostato per ignorare gli errori di compilazione

def generate(message, history):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device_map = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
	model = AutoModelForCausalLM.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens", device_map=device_map, torch_dtype=torch.bfloat16) # torch_dtype="auto", trust_remote_code=True
	model = model.to(device)

	# Esempio con chat
	# messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
	# messages.extend(history)
	# messages.append({"role": "user", "content": message})
	# messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	# model_inputs = tokenizer([messages], return_tensors="pt").to(device)
	# input_ids = model_inputs["input_ids"]
	# output = model.generate(input_ids, max_length=100, do_sample=False)

	# Esempio con codifica e decodifica
	input_ids = tokenizer.encode(message, return_tensors="pt").to(device)
	output = model.generate(input_ids, max_length=100, do_sample=False)
	generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

	# messages.extend(history)
	# messages.append({"role": "assistant", "content": generated_text})

	print(generated_text)
	return generated_text
