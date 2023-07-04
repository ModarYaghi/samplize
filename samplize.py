import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Assuming that 'df' is your DataFrame and it has columns 'Location', 'Gender', 'Age'
df = pd.read_csv('dataset423.csv')  # replace 'your_file.csv' with your actual file name

# Define the age groups and their labels
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

# Create 'Age_group' column
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Combine the stratifying columns into one
df['combined'] = df['location'].astype(str) + df['sex'].astype(str) + df['age_group'].astype(str)

# Initialize 'sample' column with 0s
df['sample'] = 0

# Get the counts for each group in 'combined'
counts = df['combined'].value_counts()

# Prepare the subset of data that can be split (i.e., each group has more than one record)
df_split = df[df['combined'].isin(counts[counts > 1].index)]

split = StratifiedShuffleSplit(n_splits=1, test_size=63, random_state=42)  # replace 63 with the size of the sample
# you want

for train_index, test_index in split.split(df_split, df_split['combined']):
    df.loc[df_split.iloc[test_index].index, 'sample'] = 1

# 'df' now has a 'sample' column indicating whether each row was selected (1) or not (0)
print(df)

df.to_csv('updated_dataset.csv', index=False, encoding='utf-8-sig')
