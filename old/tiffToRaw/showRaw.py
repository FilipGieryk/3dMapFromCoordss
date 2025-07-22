import numpy as np
import matplotlib.pyplot as plt

def show_raw_grayscale_uint16(raw_path, width, height):
    # Read the raw file as uint16
    with open(raw_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16, count=width*height)
    if data.size != width*height:
        raise ValueError("File size does not match width*height")
    # Reshape to 2D image
    img = data.reshape((height, width))
    # Optionally normalize for display (matplotlib expects 0-255 or 0-1 for grayscale)
    img_display = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize to 0-1
    # Show the image
    plt.imshow(img_display, cmap='gray')
    plt.title(f"{raw_path} ({width}x{height})")
    plt.axis('off')
    plt.show()

# Example usage:
show_raw_grayscale_uint16('resized.raw', 513, 513)
#  1737, Height: 1533