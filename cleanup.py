import json
import random

data = []

with open('data/war_wq.json') as f:
    data += json.load(f)
with open('data/nowar_wq.json') as f:
    data += json.load(f)

random.shuffle(data)
with open('data/complete.json','w') as f:
    json.dump(data, f, indent=True)
