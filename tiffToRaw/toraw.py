import rasterio
import numpy as np

def geotiff_to_grayscale_raw(input_tif, output_raw):
    with rasterio.open(input_tif) as src:
        # Read all bands
        data = src.read()
        # If the image is already single-band, just use it
        if data.shape[0] == 1:
            gray = data[0]
        else:
            # Assume RGB, convert to grayscale using standard weights
            r = data[0].astype(np.float32)
            g = data[1].astype(np.float32)
            b = data[2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        # Normalize to 16-bit unsigned integer
        gray = np.clip(gray, 0, 65535)
        gray = gray.astype(np.uint16)
        # Write to raw file
        gray.tofile(output_raw)
        print(f"Saved RAW file: {output_raw}")
        print(f"Width: {gray.shape[1]}, Height: {gray.shape[0]} (use these in Unity)")

# Example usage:
geotiff_to_grayscale_raw('cropped.tif', 'output.raw')