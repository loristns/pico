from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

print(ds['train'][120])

with open('data/wikitext-103-v1-train.txt', 'w') as f:
    for i in range(len(ds['train'])):
        print(f'Writing {i}th line over {len(ds["train"])}')

        if i % 1000 == 0:
            print(ds['train'][i])

        f.write(ds['train'][i]['text'])
