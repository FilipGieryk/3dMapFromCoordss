import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

# Load original
raw = np.fromfile('output.raw', dtype=np.uint16).reshape((340, 320))

# Smooth to reduce aliasing
raw_smoothed = gaussian_filter(raw, sigma=1)

# Resize down
resized = resize(raw_smoothed, (513, 513), order=1, preserve_range=True)

# Normalize
resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-6) * 65535
resized = resized.astype(np.uint16)

# Save
resized.tofile('resized.raw')