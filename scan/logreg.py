import torch
from easy_transformer import EasyTransformer
import json
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

activation_cache = []
def activation_hook(neuron_acts, hook):
    activation_cache.append(neuron_acts.to('cpu'))

layer = 3
mlp_layer_size = 3072

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2').to(device)
print("Loaded model")

model.blocks[layer].mlp.hook_post.add_hook(activation_hook)
print("Added hook")
print("---------------")

with open('data/complete.json') as f:
    data = json.load(f)

n = len(data)
n_train = int(n * 0.8)
X = torch.zeros((n, mlp_layer_size))
y = torch.zeros((n,))
for i,item in enumerate(data[:n]):
    prompt = item['text']
    y[i] = item['about_war']
    activation_cache = []
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)
    if prompt_tokens.shape[1] == 0:
        continue
    model(prompt_tokens)

    if len(activation_cache) != 1:
        raise Exception(f"Expecting one activation tensor, got {len(activation_cache)}")
    if activation_cache[0].shape[0] != 1:
        raise Exception(f"Expecting one prompt, got {activation_cache[0].shape[0]}")
    activations = activation_cache[0][0]
    avg_activations = activations.mean(dim=0)
    X[i] = avg_activations
    if i % 10 == 0:
        print(f'{i} / {n}')
print(X)
print(y)

analyzer = LogisticRegression(C = 0.1)
analyzer.fit(X[:n_train,:],y[:n_train])
yguess = analyzer.predict_proba(X[n_train:,:])[:,1]
print(yguess.shape)
print(y[n_train:].shape)
plt.scatter(yguess, y[n_train:] + torch.rand(n-n_train) * 0.1)
plt.show()
print(analyzer.get_params())
