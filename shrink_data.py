import pandas as pd

df = pd.read_csv('fake_job_postings.csv')

# Keep ALL fraudulent jobs (866) + a random sample of 3000 real jobs
fraud = df[df['fraudulent'] == 1]
real_sample = df[df['fraudulent'] == 0].sample(n=3000, random_state=42)

small_df = pd.concat([fraud, real_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

small_df.to_csv('fake_job_postings.csv', index=False)
print(f"New dataset size: {len(small_df)} rows")
print(small_df['fraudulent'].value_counts())