import zipfile
import json
import os
import shutil

INPUT_MODEL = "skin_classifier_efficientnet_corrected.keras"
OUTPUT_MODEL = "skin_classifier_fixed.keras"

print("🔧 Fixing model...")

# Create temporary directory
TEMP_DIR = "_model_temp"

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR)

# Extract the .keras file
with zipfile.ZipFile(INPUT_MODEL, "r") as z:
    z.extractall(TEMP_DIR)

# Find config.json
config_path = os.path.join(TEMP_DIR, "config.json")

if not os.path.exists(config_path):
    raise FileNotFoundError("❌ config.json not found inside the model!")

# Read configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Remove unsupported quantization_config recursively
removed = 0

def remove_quantization_config(obj):
    global removed

    if isinstance(obj, dict):
        if "quantization_config" in obj:
            del obj["quantization_config"]
            removed += 1

        for value in obj.values():
            remove_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


remove_quantization_config(config)

# Save modified config
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"✅ Removed {removed} quantization_config entries.")

# Re-create fixed .keras file
with zipfile.ZipFile(
    OUTPUT_MODEL,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            archive_path = os.path.relpath(file_path, TEMP_DIR)
            z.write(file_path, archive_path)

# Clean temporary folder
shutil.rmtree(TEMP_DIR)

print("✅ Fixed model created!")
print("📁", OUTPUT_MODEL)