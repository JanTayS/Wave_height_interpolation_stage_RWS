import pandas as pd

# create a sample DataFrame
data = {'col1': ['250/10', '243/10', '234/10', '245/10', '254/10', '247/8', '236/8', '249/7', '243/10', '228/10', '240/10', '254/10', '253/10', '226/10', '235/10', '235/10', '228/10', '226/10', '213/10', '231/10']}
df = pd.DataFrame(data)

# extract quality values and count occurrences
counts = {}
for col in df.columns:
    # split values and extract quality
    quality = df[col].str.split('/').str[1]
    # count occurrences of each quality value
    counts[col] = quality.value_counts()

print(counts)