import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# resnet
"""
set manipulate percentage: 0/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0118, Accuracy: 0.6825, Time consumed:24.38s

set manipulate percentage: 1/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0205, Accuracy: 0.5175, Time consumed:24.05s

set manipulate percentage: 2/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0310, Accuracy: 0.3580, Time consumed:23.78s

set manipulate percentage: 3/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0421, Accuracy: 0.2283, Time consumed:23.98s

set manipulate percentage: 4/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0532, Accuracy: 0.1422, Time consumed:24.17s

set manipulate percentage: 5/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0620, Accuracy: 0.0980, Time consumed:24.54s
"""
"""
[Thu Sep 26 22:47:05 2024] >> checkpoint/resnet18/Wednesday_03_July_2024_10h_42m_56s/resnet18-200-regular.pth
[Thu Sep 26 22:47:05 2024] >> set manipulate percentage: 0/100%
[Thu Sep 26 22:47:33 2024] >> Test set: Epoch: 0, Average loss: 0.0117, Accuracy: 0.6825, Time consumed:27.21s
[Thu Sep 26 22:47:33 2024] >> set manipulate percentage: 0.5/100%
[Thu Sep 26 22:48:00 2024] >> Test set: Epoch: 0, Average loss: 0.0156, Accuracy: 0.6027, Time consumed:27.75s
[Thu Sep 26 22:48:00 2024] >> set manipulate percentage: 1.0/100%
[Thu Sep 26 22:48:29 2024] >> Test set: Epoch: 0, Average loss: 0.0201, Accuracy: 0.5178, Time consumed:29.06s
[Thu Sep 26 22:48:29 2024] >> set manipulate percentage: 1.5/100%
[Thu Sep 26 22:48:59 2024] >> Test set: Epoch: 0, Average loss: 0.0257, Accuracy: 0.4261, Time consumed:29.96s
[Thu Sep 26 22:48:59 2024] >> set manipulate percentage: 2.0/100%
[Thu Sep 26 22:49:30 2024] >> Test set: Epoch: 0, Average loss: 0.0312, Accuracy: 0.3582, Time consumed:30.34s
[Thu Sep 26 22:49:30 2024] >> set manipulate percentage: 2.5/100%
[Thu Sep 26 22:49:59 2024] >> Test set: Epoch: 0, Average loss: 0.0371, Accuracy: 0.2817, Time consumed:29.64s
[Thu Sep 26 22:49:59 2024] >> set manipulate percentage: 3.0/100%
[Thu Sep 26 22:50:30 2024] >> Test set: Epoch: 0, Average loss: 0.0424, Accuracy: 0.2236, Time consumed:30.21s
[Thu Sep 26 22:50:30 2024] >> set manipulate percentage: 3.5/100%
[Thu Sep 26 22:51:00 2024] >> Test set: Epoch: 0, Average loss: 0.0480, Accuracy: 0.1807, Time consumed:30.28s
[Thu Sep 26 22:51:00 2024] >> set manipulate percentage: 4.0/100%
[Thu Sep 26 22:51:30 2024] >> Test set: Epoch: 0, Average loss: 0.0535, Accuracy: 0.1442, Time consumed:30.44s
"""

# vgg11
"""
Mismatched elements: 99 / 100 (99.0%)
Greatest absolute difference: 1.0671054137073022e+23 at index (0, 99) (up to 1e-05 allowed)
Greatest relative difference: 0.6813558053493229 at index (0, 13) (up to 1e-05 allowed)
  _check_trace(
set manipulate percentage: 0/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0149, Accuracy: 0.6702, Time consumed:22.94s

set manipulate percentage: 1/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0479, Accuracy: 0.3245, Time consumed:22.78s

set manipulate percentage: 2/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0857, Accuracy: 0.1386, Time consumed:22.75s

set manipulate percentage: 3/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.1185, Accuracy: 0.0662, Time consumed:22.92s

set manipulate percentage: 4/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.1429, Accuracy: 0.0361, Time consumed:22.95s

set manipulate percentage: 5/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.1648, Accuracy: 0.0215, Time consumed:22.91s
"""
"""
[Thu Sep 26 22:47:41 2024] >> checkpoint/vgg11/Wednesday_17_July_2024_15h_07m_56s/vgg11-200-regular.pth
[Thu Sep 26 22:47:41 2024] >> set manipulate percentage: 0/100%
[Thu Sep 26 22:48:05 2024] >> Test set: Epoch: 0, Average loss: 0.0148, Accuracy: 0.6727, Time consumed:23.71s
[Thu Sep 26 22:48:05 2024] >> set manipulate percentage: 0.5/100%
[Thu Sep 26 22:48:30 2024] >> Test set: Epoch: 0, Average loss: 0.0302, Accuracy: 0.4864, Time consumed:24.61s
[Thu Sep 26 22:48:30 2024] >> set manipulate percentage: 1.0/100%
[Thu Sep 26 22:48:55 2024] >> Test set: Epoch: 0, Average loss: 0.0483, Accuracy: 0.3209, Time consumed:25.34s
[Thu Sep 26 22:48:55 2024] >> set manipulate percentage: 1.5/100%
[Thu Sep 26 22:49:21 2024] >> Test set: Epoch: 0, Average loss: 0.0668, Accuracy: 0.2065, Time consumed:25.57s
[Thu Sep 26 22:49:21 2024] >> set manipulate percentage: 2.0/100%
[Thu Sep 26 22:49:46 2024] >> Test set: Epoch: 0, Average loss: 0.0852, Accuracy: 0.1337, Time consumed:25.65s
[Thu Sep 26 22:49:46 2024] >> set manipulate percentage: 2.5/100%
[Thu Sep 26 22:50:12 2024] >> Test set: Epoch: 0, Average loss: 0.1011, Accuracy: 0.0958, Time consumed:25.34s
[Thu Sep 26 22:50:12 2024] >> set manipulate percentage: 3.0/100%
[Thu Sep 26 22:50:37 2024] >> Test set: Epoch: 0, Average loss: 0.1167, Accuracy: 0.0643, Time consumed:25.31s
[Thu Sep 26 22:50:37 2024] >> set manipulate percentage: 3.5/100%
[Thu Sep 26 22:51:02 2024] >> Test set: Epoch: 0, Average loss: 0.1305, Accuracy: 0.0469, Time consumed:25.50s
[Thu Sep 26 22:51:02 2024] >> set manipulate percentage: 4.0/100%
[Thu Sep 26 22:51:28 2024] >> Test set: Epoch: 0, Average loss: 0.1417, Accuracy: 0.0357, Time consumed:25.54s
"""

# vgg19
"""
set manipulate percentage: 0/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0140, Accuracy: 0.7125, Time consumed:41.94s

set manipulate percentage: 1/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0231, Accuracy: 0.4647, Time consumed:42.95s

set manipulate percentage: 2/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0372, Accuracy: 0.1771, Time consumed:41.94s

set manipulate percentage: 3/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0512, Accuracy: 0.0583, Time consumed:41.97s

set manipulate percentage: 4/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0605, Accuracy: 0.0251, Time consumed:41.95s

set manipulate percentage: 5/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0685, Accuracy: 0.0184, Time consumed:42.10s
"""
"""
[Thu Sep 26 22:47:56 2024] >> checkpoint/vgg19/Thursday_18_July_2024_10h_57m_31s/vgg19-200-regular.pth
[Thu Sep 26 22:47:56 2024] >> set manipulate percentage: 0/100%
[Thu Sep 26 22:48:45 2024] >> Test set: Epoch: 0, Average loss: 0.0140, Accuracy: 0.7125, Time consumed:48.77s
[Thu Sep 26 22:48:45 2024] >> set manipulate percentage: 0.5/100%
[Thu Sep 26 22:49:35 2024] >> Test set: Epoch: 0, Average loss: 0.0174, Accuracy: 0.6112, Time consumed:50.03s
[Thu Sep 26 22:49:35 2024] >> set manipulate percentage: 1.0/100%
[Thu Sep 26 22:50:24 2024] >> Test set: Epoch: 0, Average loss: 0.0229, Accuracy: 0.4650, Time consumed:49.45s
[Thu Sep 26 22:50:24 2024] >> set manipulate percentage: 1.5/100%
[Thu Sep 26 22:51:14 2024] >> Test set: Epoch: 0, Average loss: 0.0305, Accuracy: 0.2989, Time consumed:49.34s
[Thu Sep 26 22:51:14 2024] >> set manipulate percentage: 2.0/100%
[Thu Sep 26 22:52:01 2024] >> Test set: Epoch: 0, Average loss: 0.0377, Accuracy: 0.1713, Time consumed:46.95s
[Thu Sep 26 22:52:01 2024] >> set manipulate percentage: 2.5/100%
[Thu Sep 26 22:52:46 2024] >> Test set: Epoch: 0, Average loss: 0.0447, Accuracy: 0.0963, Time consumed:45.59s
[Thu Sep 26 22:52:46 2024] >> set manipulate percentage: 3.0/100%
[Thu Sep 26 22:53:32 2024] >> Test set: Epoch: 0, Average loss: 0.0512, Accuracy: 0.0588, Time consumed:45.74s
[Thu Sep 26 22:53:32 2024] >> set manipulate percentage: 3.5/100%
[Thu Sep 26 22:54:18 2024] >> Test set: Epoch: 0, Average loss: 0.0565, Accuracy: 0.0378, Time consumed:46.28s
[Thu Sep 26 22:54:18 2024] >> set manipulate percentage: 4.0/100%
[Thu Sep 26 22:55:04 2024] >> Test set: Epoch: 0, Average loss: 0.0610, Accuracy: 0.0241, Time consumed:45.76s
"""


# squeezenet
"""
Test set: Epoch: 0, Average loss: 0.0103, Accuracy: 0.6783, Time consumed:48.53s

set manipulate percentage: 1/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0320, Accuracy: 0.2955, Time consumed:48.18s

set manipulate percentage: 2/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0603, Accuracy: 0.1022, Time consumed:48.44s

set manipulate percentage: 3/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0775, Accuracy: 0.0460, Time consumed:48.34s

set manipulate percentage: 4/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0896, Accuracy: 0.0267, Time consumed:48.38s

set manipulate percentage: 5/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0981, Accuracy: 0.0197, Time consumed:48.40s
"""
"""
[Thu Sep 26 22:48:14 2024] >> checkpoint/squeezenet/Thursday_18_July_2024_14h_14m_23s/squeezenet-200-regular.pth
[Thu Sep 26 22:48:14 2024] >> set manipulate percentage: 0/100%
[Thu Sep 26 22:49:12 2024] >> Test set: Epoch: 0, Average loss: 0.0103, Accuracy: Time consumed:57.56s
[Thu Sep 26 22:49:12 2024] >> set manipulate percentage: 0.5/100%
[Thu Sep 26 22:50:10 2024] >> Test set: Epoch: 0, Average loss: 0.0172, Accuracy: Time consumed:58.27s
[Thu Sep 26 22:50:10 2024] >> set manipulate percentage: 1.0/100%
[Thu Sep 26 22:51:08 2024] >> Test set: Epoch: 0, Average loss: 0.0321, Accuracy: Time consumed:57.81s
[Thu Sep 26 22:51:08 2024] >> set manipulate percentage: 1.5/100%
[Thu Sep 26 22:52:03 2024] >> Test set: Epoch: 0, Average loss: 0.0476, Accuracy: Time consumed:55.16s
[Thu Sep 26 22:52:03 2024] >> set manipulate percentage: 2.0/100%
[Thu Sep 26 22:52:56 2024] >> Test set: Epoch: 0, Average loss: 0.0605, Accuracy: Time consumed:53.08s
[Thu Sep 26 22:52:56 2024] >> set manipulate percentage: 2.5/100%
[Thu Sep 26 22:53:49 2024] >> Test set: Epoch: 0, Average loss: 0.0704, Accuracy: Time consumed:52.91s
[Thu Sep 26 22:53:49 2024] >> set manipulate percentage: 3.0/100%
[Thu Sep 26 22:54:42 2024] >> Test set: Epoch: 0, Average loss: 0.0781, Accuracy: Time consumed:52.73s
[Thu Sep 26 22:54:42 2024] >> set manipulate percentage: 3.5/100%
[Thu Sep 26 22:55:34 2024] >> Test set: Epoch: 0, Average loss: 0.0842, Accuracy: Time consumed:52.72s
[Thu Sep 26 22:55:34 2024] >> set manipulate percentage: 4.0/100%
[Thu Sep 26 22:56:26 2024] >> Test set: Epoch: 0, Average loss: 0.0896, Accuracy: Time consumed:51.76s
"""

#darknet
"""
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0174, Accuracy: 0.4904, Time consumed:17.51s

set manipulate percentage: 1/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0375, Accuracy: 0.1426, Time consumed:17.34s

set manipulate percentage: 2/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0470, Accuracy: 0.0549, Time consumed:17.44s

set manipulate percentage: 3/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0531, Accuracy: 0.0265, Time consumed:17.37s

set manipulate percentage: 4/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0572, Accuracy: 0.0170, Time consumed:17.29s

set manipulate percentage: 5/100%
Evaluating Network.....
Test set: Epoch: 0, Average loss: 0.0612, Accuracy: 0.0133, Time consumed:17.46s
"""
"""
[Thu Sep 26 22:48:27 2024] >> checkpoint/darknet/Thursday_18_July_2024_16h_28m_01s/darknet-200-regular.pth
[Thu Sep 26 22:48:27 2024] >> set manipulate percentage: 0/100%
[Thu Sep 26 22:48:48 2024] >> Test set: Epoch: 0, Average loss: 0.0174, Accuracy: 0.4904, Time consumed:21.30s
[Thu Sep 26 22:48:48 2024] >> set manipulate percentage: 0.5/100%
[Thu Sep 26 22:49:09 2024] >> Test set: Epoch: 0, Average loss: 0.0291, Accuracy: 0.2686, Time consumed:21.03s
[Thu Sep 26 22:49:09 2024] >> set manipulate percentage: 1.0/100%
[Thu Sep 26 22:49:30 2024] >> Test set: Epoch: 0, Average loss: 0.0372, Accuracy: 0.1495, Time consumed:21.02s
[Thu Sep 26 22:49:30 2024] >> set manipulate percentage: 1.5/100%
[Thu Sep 26 22:49:52 2024] >> Test set: Epoch: 0, Average loss: 0.0429, Accuracy: 0.0809, Time consumed:21.45s
[Thu Sep 26 22:49:52 2024] >> set manipulate percentage: 2.0/100%
[Thu Sep 26 22:50:13 2024] >> Test set: Epoch: 0, Average loss: 0.0473, Accuracy: 0.0503, Time consumed:21.43s
[Thu Sep 26 22:50:13 2024] >> set manipulate percentage: 2.5/100%
[Thu Sep 26 22:50:34 2024] >> Test set: Epoch: 0, Average loss: 0.0506, Accuracy: 0.0361, Time consumed:21.25s
[Thu Sep 26 22:50:34 2024] >> set manipulate percentage: 3.0/100%
[Thu Sep 26 22:50:55 2024] >> Test set: Epoch: 0, Average loss: 0.0528, Accuracy: 0.0276, Time consumed:21.05s
[Thu Sep 26 22:50:55 2024] >> set manipulate percentage: 3.5/100%
[Thu Sep 26 22:51:17 2024] >> Test set: Epoch: 0, Average loss: 0.0550, Accuracy: 0.0212, Time consumed:21.41s
[Thu Sep 26 22:51:17 2024] >> set manipulate percentage: 4.0/100%
[Thu Sep 26 22:51:38 2024] >> Test set: Epoch: 0, Average loss: 0.0573, Accuracy: 0.0169, Time consumed:20.65s
"""

manipulate_percentage = [0, 1, 2, 3, 4]
resnet18_acc = [0.6825, 0.5175, 0.3580, 0.2283, 0.1422]
vgg11_acc = [0.6702, 0.3245, 0.1386, 0.0662, 0.0361]
vgg19_acc = [0.7125, 0.4647, 0.1771, 0.0583, 0.0251]
squeezenet_acc = [0.6783, 0.2955, 0.1022, 0.0460, 0.0267]
darknet_acc = [0.4904, 0.1426, 0.0549, 0.0265, 0.0170]


manipulate_percentage = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
resnet18_acc = [0.6825, 0.6027, 0.5178, 0.4261, 0.3582, 0.2817, 0.2236, 0.1807, 0.1442, ]
vgg11_acc = [0.6727, 0.4864, 0.3209, 0.2065, 0.1337, 0.0958, 0.0643, 0.0469, 0.0357, ]
vgg19_acc = [0.7125, 0.6112, 0.4650, 0.2989, 0.1713, 0.0963, 0.0588, 0.0378, 0.0241, ]
squeezenet_acc = [0.6783, 0.5129, 0.3021, 0.1633, 0.0962, 0.0628, 0.0417, 0.0329, 0.0234, ]
darknet_acc = [0.4904, 0.2686, 0.1495, 0.0809, 0.0503, 0.0361, 0.0276, 0.0212, 0.0169, ]



datasets = [
    ('Darknet', darknet_acc, 10, 0.05),
    ('Squeeze', squeezenet_acc, 15, 0.05),
    ('ResNet', resnet18_acc, 40, 0.05),
    ('VGG11', vgg11_acc, 15, -0.02),
    ('VGG19', vgg19_acc, 25, 0.05),
]

plt.figure(figsize=(13, 10))

for label, data, week_50p, yp in datasets:
    # Plot the lines
    plt.plot(manipulate_percentage, data, linewidth=5, label=label)

    # Find half of the initial value
    half_value = data[0] / 2

    # Interpolate between the points where the value crosses half_value
    try:
        # Find the index where the data crosses or falls below half the initial value
        idx_above = np.where(np.array(data) > half_value)[0][-1]  # Last index above half_value
        idx_below = np.where(np.array(data) <= half_value)[0][0]  # First index below half_value

        # Get the corresponding x and y values for interpolation
        x_above = manipulate_percentage[idx_above]
        y_above = data[idx_above]
        x_below = manipulate_percentage[idx_below]
        y_below = data[idx_below]

        # Linear interpolation to find the x-value where y reaches half_value
        interpolated_x = x_above + (half_value - y_above) * (x_below - x_above) / (y_below - y_above)

        # Plot the 'x' marker at the interpolated point
        plt.plot(interpolated_x, half_value, '*', markersize=20, markeredgewidth=3, color=plt.gca().lines[-1].get_color())
        # Add text next to the 'x' marker
        plt.text(interpolated_x - 0.2, half_value-yp, f't={week_50p}w', fontweight='bold', fontsize=26)
    except:
        pass

plt.xlabel('Number of Errors (%)', fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')

plt.ylabel('Network Accuracy (CIFAR100)', fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')

# plt.title('Title', fontsize=16, fontweight='bold')
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
# plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
plt.grid(True)
plt.legend(fontsize=28)
plt.savefig('_cifar100.png', dpi=800, bbox_inches='tight')

plt.show()