import os
import shutil

# --- CONFIGURATION ---
source_folder = "D:/Jeremy/UoN/Year 4/Final Year Project Coding/FYP_BaselineNN/datasets/FVP-Baseline.v1-v3.yolov8" # Path to unzipped folder
output_folder = "D:/Jeremy/UoN/Year 4/Final Year Project Coding/FYP_BaselineNN/datasets/Minority_Boost"
# Replace these with the actual IDs of Part 1 and Part 5 (Check your data.yaml)
target_ids = ["0", "4"] 

# Create output folders
os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

label_path = os.path.join(source_folder, "train/labels")
image_path = os.path.join(source_folder, "train/images")

for label_file in os.listdir(label_path):
    with open(os.path.join(label_path, label_file), 'r') as f:
        lines = f.readlines()
        
    # Check if the file contains Part 1 or Part 5
    for line in lines:
        class_id = line.split()[0]
        if class_id in target_ids:
            # Copy Label
            shutil.copy(os.path.join(label_path, label_file), os.path.join(output_folder, "labels"))
            # Copy Image (assuming .jpg, change if .png)
            img_name = label_file.replace(".txt", ".jpg")
            shutil.copy(os.path.join(image_path, img_name), os.path.join(output_folder, "images"))
            break # Move to next file once a match is found

print(f"Done! Check your folder at: {output_folder}")