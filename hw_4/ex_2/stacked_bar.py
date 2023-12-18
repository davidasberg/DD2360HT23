import matplotlib.pyplot as plt
import numpy as np

copy_to_device = [1.90000, 2, 3, 4]  # replace with your actual values
kernel = [0.456000, 3, 4, 5]  # replace with your actual values
copy_to_host = [1.060000, 4, 5, 6]  # replace with your actual values

labels = ['Non-streamed\nNon-async', 'Streamed Async\nS_SEG=1', 'Streamed Async\nS_SEG=2', 'Streamed Async\nS_SEG=4']

fig, ax = plt.subplots()

# Create the 'copy_to_device' bars
ax.bar(labels, copy_to_device, label='Copy to Device')

# Stack 'kernel' bars on top, by specifying the bottom to be the top of 'copy_to_device' bars
ax.bar(labels, kernel, bottom=copy_to_device, label='Kernel')

# Similarly, stack 'copy_to_host' bars on top of both
total_bottom = np.add(copy_to_device, kernel)
ax.bar(labels, copy_to_host, bottom=total_bottom, label='Copy to Host')

ax.set_ylabel('Time (ms)')
ax.set_title('Execution Time for vecAdd Kernel (1 000 000 elements)')
ax.legend()

plt.show()