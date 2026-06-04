import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# 1. Define distinct data levels (must have one more boundary than colors)
levels = [-20, -10, 0, 10, 20]

# 2. Define your distinct color palette (no gradients)
color_list = ['#FF5733', '#33FF57', '#3357FF', '#F033FF']

# 3. Create the colormap and normalization
cmap = colors.ListedColormap(color_list)
norm = colors.BoundaryNorm(levels, cmap.N)

# 4. Generate dummy data
data = np.random.uniform(-15, 15, size=(10, 10))

# 5. Plot with the defined palette without gradient
plt.imshow(data, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()
