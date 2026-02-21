import os

base_dir = r"D:\M3 Projects\DeepFake_Research"
splits_dir = os.path.join(base_dir, "outputs", "splits")
results_dir = os.path.join(base_dir, "outputs", "results")

old_prefix = r"D:\M3\DeepFake_Research"
new_prefix = r"D:\M3 Projects\DeepFake_Research"

def fix_csv(file_path):
    print(f"Checking {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if old_prefix in content:
        print(f"  Fixing paths in {os.path.basename(file_path)}")
        new_content = content.replace(old_prefix, new_prefix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        print(f"  No incorrect paths found in {os.path.basename(file_path)}")

# Fix splits
for f in os.listdir(splits_dir):
    if f.endswith(".csv"):
        fix_csv(os.path.join(splits_dir, f))

# Fix results
for f in os.listdir(results_dir):
    if f.endswith(".csv"):
        fix_csv(os.path.join(results_dir, f))
