import torch
from easy_transformer import EasyTransformer
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

short = 10

layer = 11
mlp_layer_size = 3072
vocab_size = 50257
num_to_display = 20

activation_cache = []
def activation_hook(neuron_acts, hook):
    activation_cache.append(neuron_acts.to('cpu'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2').to(device)
print("Loaded model. n_blocks = {len(model.blocks)}")

#model.blocks[layer].mlp.hook_post.add_hook(activation_hook)
model.blocks[layer].hook_mlp_out.add_hook(activation_hook)
print("Added hook")

war_index = model.tokenizer.encode(' war')[0]
print(f'War index: {war_index}')

print("---------------")
with open('data/complete.json') as f:
    data = json.load(f)

n = len(data)
if short != None:
    n = min(short, n)
n_train = int(n * 0.8)
X = torch.zeros((n, mlp_layer_size))
y = torch.zeros((n,))
tokens = [None] * n
probs = [None] * n
activs = [None] * n
token_n_train = 0
for i,item in enumerate(data[:n]):
    prompt = item['text']
    y[i] = item['about_war']
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
    if prompt_tokens.shape[1] == 0:
        continue
    tokens[i] = prompt_tokens[0]
    activation_cache = []
    logits = model(prompt_tokens)[0,:,:].to('cpu')
    probs[i] = torch.nn.functional.softmax(logits, dim=1)
    activs[i] = activation_cache[0][0]
    if i < n_train:
        token_n_train += prompt_tokens.shape[1]
    if i % 10 == 0:
        print(f'{i} / {n}')
print(f'Got probs and activations. {token_n_train} total training tokens')

unembed = model.unembed.W_U.data.to('cpu')

X = []
y = []
for i in range(n_train, min(n + num_to_display, n)):
    if tokens[i] == None:
        continue
    string = ''

    activ = activs[i]
    print(activ.shape, unembed.shape)
    guess_logits = torch.matmul(activ, unembed)
    guess_probs = torch.nn.functional.softmax(guess_logits, dim=1)
    for j in range(len(tokens[i])):
        tok = model.tokenizer.decode(tokens[i][j])
        if j > 0:
            prob = probs[i][j-1,war_index].item()

            if prob > 0.001:
                X.append(
                y.append(prob)
                string += '\033[31m'
                string += tok
            elif prob > 0.001:
                string += '\033[33m'
                string += tok
            else:
                string += '\033[m'
                string += tok
        else:
            string += '\033[m'
            string += tok
        print(f'{string}\033[m')

