"""Extract the dataset zip, skipping any corrupted files."""
import zipfile
import os

zip_path = 'artifacts/raw_data/cat-and-dog.zip'
dest = 'artifacts/raw_data'

z = zipfile.ZipFile(zip_path)
entries = z.namelist()
print(f"Total entries in zip: {len(entries)}")

extracted = 0
skipped = 0
errors = []

for entry in entries:
    try:
        z.extract(entry, dest)
        extracted += 1
    except Exception as e:
        skipped += 1
        errors.append(f"  {entry}: {e}")

z.close()
print(f"Extracted: {extracted}")
print(f"Skipped (corrupted): {skipped}")
if errors:
    print("Errors:")
    for err in errors[:10]:
        print(err)

# Count final images
cat_count = 0
dog_count = 0
for root, dirs, files in os.walk(dest):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            fname = f.lower()
            parent = os.path.basename(root).lower()
            if fname.startswith('cat') or 'cat' in parent:
                cat_count += 1
            elif fname.startswith('dog') or 'dog' in parent:
                dog_count += 1

print(f"\nFinal count: {cat_count} cat images, {dog_count} dog images")
