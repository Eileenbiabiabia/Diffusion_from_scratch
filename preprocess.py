import os
from PIL import Image
from torchvision import transforms
import torch
import multiprocessing as mp
from tqdm import tqdm

# Name the file name to store preprocessed images
raw_dir = './data/img_align_celeba'
save_dir = './data/celeba_tensor64'
os.makedirs(save_dir, exist_ok=True)

# I want to center crop the image to 178x178 and then resize it to 64x64
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

def process_and_save(idx_filename):
    idx, filename = idx_filename
    try:
        img_path = os.path.join(raw_dir, filename)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        torch.save(tensor, os.path.join(save_dir, f"{idx:06d}.pt"))
    except Exception as e:
        print(f"Error with {filename}: {e}")

image_files = sorted([
    f for f in os.listdir(raw_dir)
    if f.endswith('.jpg')
])

if __name__ == '__main__':
    #usemiltiprocessing to speed up the process
    mp.set_start_method('fork', force=True)  
    with mp.Pool(processes=8) as pool:       # I request 8 cores
        list(tqdm(pool.imap(process_and_save, enumerate(image_files)), total=len(image_files)))
    print("Resize and Save finished")