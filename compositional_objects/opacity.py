from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

blender_dir = Path.home() / "Google Drive/My Drive/Blender/compositional_objects"

path = blender_dir / "images" / "TBPLogo-standard_original.png"

# Load image and convert to RGBA numpy array
img = Image.open(path)
rgba = np.array(img.convert("RGBA"))
alpha = rgba[:, :, 3]

plt.imshow(alpha)
plt.show()
