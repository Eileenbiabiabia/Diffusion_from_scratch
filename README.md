# Diffusion-from-Scratch

This project implements diffusion-based generative models for image generation, with a focus on a UNet-based Denoising Diffusion Probabilistic Model (DDPM). The goal is to reproduce DDPM training and inference from scratch using PyTorch and apply it to datasets like CelebA.

## ✨ Features

- From-scratch implementation of DDPM with UNet
- Modular architecture: ResBlock, Attention, and sinusoidal time embeddings
- Exponential Moving Average (EMA) model for stable inference
- Image preprocessing pipeline for CelebA
- Periodic checkpoint saving
- Training loss visualization and denoising trajectory outputs

---


## 📁 Project Structure
Implementation of different diffusion models for image generation 
<pre>
```plaintext
Diffusion-Models-from-Scratch/
├── checkpoints/
├── data/
│   ├── img_align_celeba/         # Raw CelebA images
│   └── celeba_tensor64/          # Preprocessed image tensors
├── models/
│   ├── unet.py                   # UNet and supporting blocks
│   └── utils.py                  # EMA and DDPM scheduler
├── output/                       # Output images and training loss
├── output_logs                   # Logs from Quest
├── train.py                      # Training script
├── preprocess.py                 # CelebA image → tensor script
├── environment.yml              # Dependencies
└── README.md
```
</pre>
---

## 🚀 Getting Started

### Step 1: Setup

git clone https://github.com/Eileenbiabiabia/Diffusion_from_scratch.git
cd Diffusion_from_scratch
conda env create -f environment.yml
conda activate genai

### Step 2: Preprocessing

Download the CelebA dataset and place images under ./data/img_align_celeba/
    You can find the CelebA dataset here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
Preprocess images using multiprocessing: 
python preprocess.py
Or you can submit a jobscript to allocate a 8 cpu per tasks on quest to train. 
sbatch jobscript_preprocess.sh


### Step 3: Training
You can submit a jobscript to allocate a A100/H100 on quest to train. 
sbatch jobscript_train.sh

The model will:
	•	Train on 30,000 CelebA samples by default
	•	Save checkpoints every 10 epochs to checkpoints/
	•	Output loss visualization to output/
