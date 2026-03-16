import numpy as np

Y0 = (-10.0, 10.0)
y_split = 0.0

new_y1 = (Y0[0], y_split)
new_y2 = (y_split, Y0[1])

print(f"y1: {np.linspace(new_y1[0], new_y1[1], 10)}")
print(f"y2: {np.linspace(new_y2[0], new_y2[1], 10)}")
