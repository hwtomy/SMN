import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_fft_magnitude(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    return magnitude_spectrum

def compute_frequency_ratio(magnitude_spectrum, radius_ratio=0.25):
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    low_freq = magnitude_spectrum[dist <= radius_ratio * max(h, w)].sum()
    high_freq = magnitude_spectrum[dist > radius_ratio * max(h, w)].sum()

    return low_freq / (low_freq + high_freq), high_freq / (low_freq + high_freq)

def plot_fft(image_path, magnitude_spectrum, output_dir):
    os.makedirs(output_dir, exist_ok=True)  

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_fft.png")

    plt.figure(figsize=(10, 4))


    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path).convert('L'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(magnitude_spectrum), cmap='inferno')
    plt.title('Log Magnitude Spectrum')
    plt.axis('off')

    plt.suptitle(base_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # print(f"Saved FFT to: {save_path}")

def analyze_folder_fft(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            magnitude_spectrum = compute_fft_magnitude(image_path)
            plot_fft(image_path, magnitude_spectrum, output_folder)
            low_ratio, high_ratio = compute_frequency_ratio(magnitude_spectrum)
            print(f"{filename}: Low freq: {low_ratio:.2%}, High freq: {high_ratio:.2%}")

input_folder = "codak/"       
output_folder = "results/" 

analyze_folder_fft(input_folder, output_folder)