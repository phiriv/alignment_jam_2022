import torch
from easy_transformer import EasyTransformer

all_hooks_fn = lambda name: True
def print_shape(tensor, hook):
    print(f'Activation at hook {hook.name} has shape: {tensor.shape}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2').to(device)
print("Loaded model. n_blocks = {len(model.blocks)}")

random_tokens = torch.randint(1000, 10000, (4, 50))
logits = model.run_with_hooks(random_tokens, fwd_hooks=[(all_hooks_fn, print_shape)])

