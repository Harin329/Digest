import pandas as pd

df = pd.read_csv("data/full_dataset.csv") 
df = df.loc[:,"directions"]
with open('data/directions.txt', 'w') as f:
    for row in df.items():
        steps = row[1][2:-2]
        steps = steps.replace('", "', " ")
        print(len(steps))
        f.write(steps)