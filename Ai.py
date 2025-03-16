import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

try:
    with open('user_logs.json', 'r') as f:
        user_logs = json.load(f)
except FileNotFoundError:
    user_logs = []

def log_user_input(user_input):
    user_logs.append(user_input)
    with open('user_logs.json', 'w') as f:
        json.dump(user_logs, f)

def evaluate_math_expression(expression):
    regex = r'^[0-9+\-*/.() ]+$'
    if re.match(regex, expression):
        try:
            return str(eval(expression))
        except Exception:
            return "I can't compute that."
    else:
        return "That's not a valid math expression."

def get_response(user_input, chat_history_ids=None):
    log_user_input(user_input)

    if re.match(r'^\d+[\s\+\-\*/]+[0-9]+$', user_input):
        return evaluate_math_expression(user_input)

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)

    max_length = 1000
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]

    attention_mask = torch.ones(input_ids.shape, dtype=torch.float)

    chat_history_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

def chat():
    print("Chat with the AI! Type 'exit' to stop.")
    chat_history_ids = None
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chat ended.")
                break
            response, chat_history_ids = get_response(user_input, chat_history_ids)
            print("AI: " + response)
        except KeyboardInterrupt:
            print("\nChat interrupted. Exiting...")
            break

if __name__ == "__main__":
    chat()