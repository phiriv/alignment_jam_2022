import wikiquotes
import json
import os

results = []
for topic in open('random_pages.txt'):
    quotes = wikiquotes.get_quotes(topic,'english')
    for quote in quotes[:10]:
        results.append({
            'text': quote,
            'about_war': 0,
            'source': 'wikiquote'
        })

os.makedirs('data', exist_ok=True)
with open('data/nowar_wq.json','w') as f:
    json.dump(results, f, indent=True)

