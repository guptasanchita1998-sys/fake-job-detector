import pandas as pd

# Dataset load karo
df = pd.read_csv('fake_job_postings.csv')

# Kitne rows/columns hain dekho
print("Shape of dataset:", df.shape)

# Pehli 5 rows dekho
print(df.head())

# Column names dekho
print("\nColumns:", df.columns.tolist())

# Kitni fake aur kitni real jobs hain
print("\nFraudulent job count:")
print(df['fraudulent'].value_counts())