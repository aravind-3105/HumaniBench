from PIL import Image
import os
# Create the output directory if it doesn't exist
os.makedirs("/projects/NMB-Plus/E-VQA/data/eval6/empathy_dataset/resized", exist_ok=True)

# Define folders and target size
input_folder = "/projects/NMB-Plus/E-VQA/data/eval6/empathy_dataset/images"           # Folder containing your original images
output_folder = "/projects/NMB-Plus/E-VQA/data/eval6/empathy_dataset/resized"           # Folder to save resized images
target_size = (350, 350)  # Desired width and height

# Process each image in the input folder
for count, file_name in enumerate(os.listdir(input_folder)):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        input_path = os.path.join(input_folder, file_name)
        try:
            with Image.open(input_path) as img:
                # Resize the image using the new resampling filter
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                # Generate a new simple name for the image
                new_name = f"image_{count + 1}.jpg"
                output_path = os.path.join(output_folder, new_name)
                # Save the resized image
                img_resized.save(output_path)
                print(f"Processed {file_name} -> {new_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")