import pandas as pd

output_name = "process_prm800k.json"

df = pd.read_parquet('/home/yangkai/LLaMA-Factory/data/process_prm800k/train-00000-of-00001.parquet')

df['instruction'] = df['text'].str.split('# Solution\n\n').str[0].str.replace('# Question\n\n', '', regex=False)
df['output'] = df['text'].str.split('# Solution\n\n').str[1]

df['input'] = ""

# print(df["text"])
print(df[['instruction', 'output']].head())


df_result = df[['instruction', 'input', 'output']]

df_result.to_json(output_name, orient='records', force_ascii=False, indent=2)

print(f"Data has been saved to '{output_name}'")