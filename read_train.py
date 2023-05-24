import pandas as pd

df_train = pd.read_csv('data/train.csv')
# Разбиение и сохранение
chunk_size = 100000
chunks = [x for x in range(0, df_train.shape[0], chunk_size)]
for i in range(len(chunks) - 1):
    df_temp = df_train.iloc[chunks[i]:chunks[i + 1]]
    df_temp.to_parquet(f'data/train/df_train_part_{i}.parquet')
# Обработка последнего блока отдельно (в случае, если количество строк не делится на chunk_size без остатка)
df_temp = df_train.iloc[chunks[-1]:]
df_temp.to_parquet(f'data/train/df_train_part_{len(chunks) - 1}.parquet')
