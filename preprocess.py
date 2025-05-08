import os
from PIL import Image
from torchvision import transforms
import torch
import multiprocessing as mp
from tqdm import tqdm

# Directory where raw CelebA images are stored
raw_dir = './data/img_align_celeba'

# Directory where the processed tensors will be saved
save_dir = './data/celeba_tensor64'
os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

# Define a transformation pipeline:
# 1. Center-crop the image to 178x178 (to remove background)
# 2. Resize to 64x64
# 3. Convert to tensor
# 4. Normalize pixel values to [-1, 1] range (mean=0.5, std=0.5 per channel)
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# Function to apply transformation and save as .pt tensor
def process_and_save(idx_filename):
    idx, filename = idx_filename
    try:
        img_path = os.path.join(raw_dir, filename)  # Full image path
        img = Image.open(img_path).convert("RGB")   # Load and convert image to RGB
        tensor = transform(img)                     # Apply defined transformations
        torch.save(tensor, os.path.join(save_dir, f"{idx:06d}.pt"))  # Save with padded filename (e.g., 000123.pt)
    except Exception as e:
        print(f"Error with {filename}: {e}")        # Catch and report errors gracefully

# Get list of all JPEG image filenames, sorted to preserve order
image_files = sorted([
    f for f in os.listdir(raw_dir)
    if f.endswith('.jpg')
])

if __name__ == '__main__':
    # Use multiprocessing to speed up preprocessing
    mp.set_start_method('fork', force=True)  
    with mp.Pool(processes=8) as pool:       # Create a pool with 8 worker processes
        list(tqdm(pool.imap(process_and_save, enumerate(image_files)), total=len(image_files)))
    print("Resize and Save finished")      