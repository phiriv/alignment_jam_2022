import torch
from easy_transformer import EasyTransformer
import sys

embed_dim = int(sys.argv[1])

device = 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2').to(device)
print("Loaded model. n_blocks = {len(model.blocks)}")

vec = model.unembed.W_U.data[embed_dim, :]
values = [(v.item(),i) for i,v in enumerate(vec)]
values.sort(reverse=True)
for i in range(len(values)):
    value, index = values[i]
    tok = model.tokenizer.decode(index)
    if i < 50 or i > len(values) - 30 or tok == ' war':
        print(i, tok, value)
