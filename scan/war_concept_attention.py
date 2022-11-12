import torch
from easy_transformer import EasyTransformer
import json
import numpy as np
import einops
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

short = 100

n_layers = 12
mlp_layer_size = 3072
vocab_size = 50257
num_to_display = 2
max_display_tokens = 20

attention_cache = [None] * n_layers
activation_cache = [None] * n_layers

def attention_hook(neuron_acts, hook, layer):
    attention_cache[layer] = neuron_acts[0,:,:,:].to('cpu')

def activation_hook(neuron_acts, hook, layer):
    activation_cache[layer] = neuron_acts[0,:,:].to('cpu')

def layer_norm_pre(x):
    global model
    x = x - x.mean(axis=-1, keepdim=True)
    scale = (einops.reduce(x.pow(2), 'pos embed -> pos 1', 'mean') + model.cfg.eps).sqrt()
    return x / scale


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2').to(device)
n_heads = model.cfg.n_heads
print(f"Loaded model. n_blocks = {len(model.blocks)}. n_heads = {n_heads}")

for layer in range(n_layers):
    model.blocks[layer].hook_resid_post.add_hook(lambda a,hook,layer=layer: activation_hook(a,hook,layer))
    model.blocks[layer].attn.hook_attn.add_hook(lambda a,hook,layer=layer: attention_hook(a,hook,layer))
print("Added hooks")

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
attens = [None] * n
token_n_train = 0
for i,item in enumerate(data[:n]):
    prompt = item['text']
    y[i] = item['about_war']
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
    if prompt_tokens.shape[1] == 0:
        continue
    tokens[i] = prompt_tokens[0]
    attention_cache = [None] * n_layers
    activation_cache = [None] * n_layers
    logits = model(prompt_tokens)[0,:,:].to('cpu')
    probs[i] = torch.nn.functional.softmax(logits, dim=1)
    activs[i] = list(activation_cache)
    attens[i] = list(attention_cache)
    if i < n_train:
        token_n_train += prompt_tokens.shape[1]
    if i % 10 == 0:
        print(f'{i} / {n}')
print(f'Got probs and activations. {token_n_train} total training tokens')

unembed_w = model.unembed.W_U.data.to('cpu')
unembed_b = model.unembed.b_U.data.to('cpu')

def pad(tok):
    if len(tok) >= 5:
        return tok[:8]
    else:
        return (' ' * (8 - len(tok))) + tok

for i in range(n_train, min(n_train + num_to_display, n)):
    if tokens[i] == None:
        continue
    n_tok = min(len(tokens[i]), max_display_tokens)

    for layer in range(n_layers):
        string = ''
        a = activs[i][layer]
        a = layer_norm_pre(a)
        guess_logits = torch.matmul(a, unembed_w) + unembed_b
        guess_probs = torch.nn.functional.softmax(guess_logits, dim=1)
        for j in range(n_tok):
            tok = model.tokenizer.decode(tokens[i][j])
            if j > 0:
                prob = guess_probs[j-1,war_index].item()

                if prob > 0.01:
                    string += '\033[31m'
                elif prob > 0.001:
                    string += '\033[33m'
                else:
                    string += '\033[m'
            else:
                string += '\033[m'
            string += pad(tok)
        print(f'{string}\033[m')
        for head in range(n_heads):
            string = ''
            for j in range(n_tok):
                #tok = model.tokenizer.decode(tokens[i][j])
                a = attens[i][layer][head, j, :]
                index = torch.argmax(a).item()
                if index == 0:
                    prevtok = '-'
                else:
                    prevtok = model.tokenizer.decode(tokens[i][index])
                string += pad(prevtok)
            print(string)

