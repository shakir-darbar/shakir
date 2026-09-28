import os
import glob
import pandas as pd

df = pd.read_csv('data/raw/diagnosis.csv', header=None, names=['pid', 'diag'])
mapping = dict(zip(df['pid'], df['diag']))

target = ['Healthy', 'Pneumonia', 'URTI', 'Bronchiectasis']
samples = {c: [] for c in target}

for f in glob.glob('data/raw/*.wav'):
    filename = os.path.basename(f)
    pid_str = filename.split('_')[0]
    if pid_str.isdigit():
        pid = int(pid_str)
        d = mapping.get(pid)
        if d in samples and len(samples[d]) < 3:
            samples[d].append(filename)

print("="*60)
print("      SAMPLE AUDIO FILES TO TEST FOR EACH DISEASE CLASS")
print("="*60)
for cls in target:
    print(f"\n--- {cls.upper()} ---")
    for fn in samples[cls]:
        print(f"  * {fn}")
print("="*60)
