import json

json_list = []

for filename in ['submit.json', 'tagging_5k_A fix_bug.json']:
    with open(filename, 'r', encoding='utf-8') as f:
        json_list.append(json.load(f))

print('in')
