import wikiquotes
import json
import os

results = []
quotes = wikiquotes.get_quotes('war','english')
for quote in quotes:
    results.append({
        'text': quote,
        'about_war': 1,
        'source': 'wikiquote'
    })

os.makedirs('data', exist_ok=True)
with open('data/war_wq.json','w') as f:
    json.dump(results, f, indent=True)

