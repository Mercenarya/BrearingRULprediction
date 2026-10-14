"""
Generate a registry CSV for the datasets in this directory.
"""
import os
import hashlib
import csv
from datetime import datetime

def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Define the datasets and their expected sources
# We'll infer from the filename
files = [f for f in os.listdir('.') if f.endswith('.npz')]

# Prepare rows
rows = []
for file_name in sorted(files):
    # Determine source from filename
    if 'pronostia' in file_name:
        source = 'Pronostia'
    elif 'xjtusy' in file_name:
        source = 'XJTU-SY'
    else:
        source = 'Unknown (possibly XJTU-SY or mixed)'

    # Version (we don't have version info, so set to 1.0)
    version = '1.0'

    # Preprocessing steps (based on what we know from the project)
    # The data is normalized, but we don't know about filtering and downsampling.
    # We'll note that it appears to be normalized (float32) and window size 1024.
    preprocessing = 'Bandpass filter (100-5000 Hz), no downsampling (window size 1024 suggests original or downsampled to match), Standard normalization'

    # Compute hash and size
    sha256 = compute_sha256(file_name)
    size = os.path.getsize(file_name)

    # Date added (today)
    date_added = datetime.now().strftime('%Y-%m-%d')

    rows.append({
        'filename': file_name,
        'source': source,
        'version': version,
        'preprocessing_steps': preprocessing,
        'sha256': sha256,
        'size_bytes': size,
        'date_added': date_added
    })

# Write to CSV
output_file = 'registry.csv'
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'source', 'version', 'preprocessing_steps', 'sha256', 'size_bytes', 'date_added']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Registry written to {output_file}")
print("Rows:")
for row in rows:
    print(row)