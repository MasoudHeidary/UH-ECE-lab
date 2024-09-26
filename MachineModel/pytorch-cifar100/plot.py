import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

resnet18_acc = [0.6825, 0.5175, 0.3580, 0.2283, 0.1422]
vgg11_acc = [0.6702, 0.3245, 0.1386, 0.0662, 0.0361]
vgg19_acc = [0.7125, 0.4647, 0.1771, 0.0583, 0.0251]
squeezenet_acc = [0.6783, 0.2955, 0.1022, 0.0460, 0.0267]
darknet_acc = [0.4904, 0.1426, 0.0549, 0.0265, 0.0170]

manipulate_percentage = [0, 1, 2, 3, 4]

if True:
    plt.figure(figsize=(13, 10))

    plt.plot(manipulate_percentage, darknet_acc, linewidth=5, label="Darknet")
    plt.plot(manipulate_percentage, squeezenet_acc, linewidth=5, label="Squeeze")
    plt.plot(manipulate_percentage, resnet18_acc, linewidth=5, label="ResNet")
    plt.plot(manipulate_percentage, vgg11_acc, linewidth=5, label="VGG11")
    plt.plot(manipulate_percentage, vgg19_acc, linewidth=5, label="VGG19")


    plt.xlabel('Number of Errors (%)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28, fontweight='bold')

    plt.ylabel('Network Accuracy (CIFAR100)', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    # plt.title('Title', fontsize=16, fontweight='bold')
    # plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    # plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    plt.grid(True)
    plt.legend(fontsize=28)
    plt.show()