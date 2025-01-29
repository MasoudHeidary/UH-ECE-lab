import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from random import random

# network Cipher10 [epoch:100], using x2 delta
# manipulate_percentage = [0, 1, 2, 3, 4]
# darknetMani_delta2 = [0.8358, 0.5617, 0.2006, 0.1135, 0.1034]
# squeezenetMani_delta2 = [0.8397, 0.2692, 0.1251, 0.1053, 0.1010]
# resnet18Mani_delta2 = [0.8666, 0.7486, 0.5593, 0.3686, 0.2440]
# vgg11Mani_delta2 = [0.8614, 0.7226, 0.4991, 0.3071, 0.2055]
# vgg19Mani_delta2 = [0.8891, 0.776, 0.4828, 0.1949, 0.1184]

manipulate_percentage = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
darknetMani_delta2 = [0.8358, 0.7189, 0.5591, 0.3609, 0.2065, 0.1379, 0.1078, 0.1049, 0.0990,]
squeezenetMani_delta2 = [0.8397, 0.5597, 0.2749, 0.1614, 0.1181, 0.1100, 0.1043, 0.0969, 0.1020,]
resnet18Mani_delta2 = [0.8666, 0.8186, 0.7451, 0.6595, 0.5584, 0.4581, 0.3658, 0.2899, 0.2390,]
vgg11Mani_delta2 = [0.8603, 0.8016, 0.7155, 0.6136, 0.4975, 0.3998, 0.3088, 0.2544, 0.2026,]
vgg19Mani_delta2 = [0.8891, 0.8460, 0.7771, 0.6482, 0.4825, 0.3105, 0.1954, 0.1489, 0.1155,]
transformer = [90.4, 50.1125, 50.1425, 50.1425, 50.63, 50.63, 50.175, 50.175, 49.83]
transformer = [i/100 for i in transformer]

datasets = [
    ('Darknet', darknetMani_delta2, 25, -0.03),
    ('Squeeze', squeezenetMani_delta2, 18, 0.05),
    ('ResNet', resnet18Mani_delta2, 62, 0.06),
    ('VGG11', vgg11Mani_delta2, 43, -0.03),
    ('VGG19', vgg19Mani_delta2, 38, 0.10),
    ('Transformer', transformer, 15, 0.05),
]
plt.figure(figsize=(13, 10))

for label, data, week_50p, yp in datasets:
    # Plot the lines
    plt.plot(manipulate_percentage, data, linewidth=8, label=label)

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
        plt.plot(interpolated_x, half_value, '*', markersize=30, markeredgewidth=5, color=plt.gca().lines[-1].get_color())
        # Add text next to the 'x' marker
        plt.text(interpolated_x - 0.2, half_value-yp, f't={week_50p}w', fontweight='bold', fontsize=30)
    except:
        pass

    if label == "Transformer":
        plt.plot(0.5, 0.5, '*', markersize=30, markeredgewidth=5, color=plt.gca().lines[-1].get_color())
        plt.text(0.5 - 0.1, 0.5-yp, f't={week_50p}w', fontweight='bold', fontsize=30, color='black')


# Set labels and styling
plt.xlabel('Number of Errors (%)', fontsize=36, fontweight='bold')
plt.xticks(fontsize=36, fontweight='bold')

plt.ylabel('Network Accuracy (CIFAR10)', fontsize=36, fontweight='bold')
plt.yticks(fontsize=36, fontweight='bold')

plt.grid(True)
plt.legend(fontsize=28)

plt.savefig('_cifar10.png', dpi=2000, bbox_inches='tight')

plt.show()



# SVHN, using regular (x1) delta
manipulate_percentage = [0, 1, 2, 3, 4]
darknet_SVHN = [0.8575, 0.6707, 0.3628, 0.1934, 0.1696]
squeezenet_SVHN = [0.8726, 0.4841, 0.2041, 0.1520, 0.1504]
resnet18_SVHN = [0.8452, 0.7469, 0.5632, 0.3660, 0.2454]
vgg11_SVHN = [0.8541, 0.6541, 0.3893, 0.2393, 0.1759]
vgg19_SVHN = [0.8638, 0.7722, 0.5138, 0.2406, 0.1543]


manipulate_percentage = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
darknet_SVHN = [0.8575, 0.7900, 0.6705, 0.5199, 0.3622, 0.2478, 0.1968, 0.1751, 0.1698, ]
squeezenet_SVHN = [0.8726, 0.7344, 0.4856, 0.2908, 0.2050, 0.1632, 0.1532, 0.1506, 0.1479, ]
resnet18_SVHN = [0.8452, 0.8075, 0.7480, 0.6601, 0.5587, 0.4572, 0.3630, 0.2968, 0.2489, ]
vgg11_SVHN = [0.8546, 0.7751, 0.6527, 0.5114, 0.3849, 0.3007, 0.2349, 0.2019, 0.1726, ]
vgg19_SVHN = [0.8638, 0.8325, 0.7724, 0.6663, 0.5131, 0.3582, 0.2421, 0.1780, 0.1520, ]


datasets = [
    ('Darknet', darknet_SVHN, 30, 0.07),
    ('Squeeze', squeezenet_SVHN, 25, 0.05),
    ('ResNet', resnet18_SVHN, 60, -0.02),
    ('VGG11', vgg11_SVHN, 31, -0.02),
    ('VGG19', vgg19_SVHN, 40, 0.05),
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
        plt.text(interpolated_x - 0.1, half_value-yp, f't={week_50p}w', fontweight='bold', fontsize=26)
    except:
        pass

plt.xlabel('Number of Errors (%)', fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')

plt.ylabel('Network Accuracy (SVHN)', fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')

# plt.title('Title', fontsize=16, fontweight='bold')
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
# plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
plt.grid(True)
plt.legend(fontsize=28)
# plt.gca().set_aspect(5)
plt.savefig('_svhn.png', dpi=800, bbox_inches='tight')

plt.show()


# accuracy loss after 1 year (2% error rate)
print(
    (
        0.8358 - 0.2006 +\
        0.8397 - 0.1251 +\
        0.8666 - 0.5593 +\
        0.8614 - 0.4991 +\
        0.8891 - 0.4828 +\
        \
        0.8575 - 0.3628 +\
        0.8726 - 0.2041 +\
        0.8452 - 0.5632 +\
        0.8541 - 0.3893 +\
        0.8638 - 0.5138 +\
        \
        0.6825 - 0.3580 +\
        0.6702 - 0.1386 +\
        0.7125 - 0.1771 +\
        0.6783 - 0.1022 +\
        0.4904 - 0.0549
    ) / 15
)

# # accuracy loss after 1.5 year
# print(
#     (
#         0.8358 - 0.1135 +\
#         0.8397 - 0.1053 +\
#         0.8666 - 0.3686 +\
#         0.8614 - 0.3071 +\
#         0.8891 - 0.1949 +\
#         \
#         0.8575 - 0.1934 +\
#         0.8726 - 0.1520 +\
#         0.8452 - 0.3660 +\
#         0.8541 - 0.2393 +\
#         0.8638 - 0.2406 +\
#         \
#         0.6825 - 0.2283 +\
#         0.6702 - 0.0662 +\
#         0.7125 - 0.0583 +\
#         0.6783 - 0.0460 +\
#         0.4904 - 0.0265
#     ) / 15
# )


# accuracy loss after 4 years ()

print(
    (
    0.8358 - 0.1034 + \
    0.8397 - 0.1010 + \
    0.8666 - 0.2440 + \
    0.8614 - 0.2055 + \
    0.8891 - 0.1184 + \
    \
    0.8575 - 0.1696 +\
    0.8726 - 0.1504 +\
    0.8452 - 0.2454 +\
    0.8541 - 0.1759 +\
    0.8638 - 0.1543 +\
    \
    0.6825 - 0.1422 +\
    0.6702 - 0.0361 +\
    0.7125 - 0.0251 +\
    0.6783 - 0.0267 +\
    0.4904 - 0.0170 
    ) / 15
)



# max accuracy lost
print(
    max(
        [
            0.8358 - 0.1034,
            0.8397 - 0.1010,
            0.8666 - 0.2440,
            0.8614 - 0.2055,
            0.8891 - 0.1184,
            0.8575 - 0.1696,
            0.8726 - 0.1504,
            0.8452 - 0.2454,
            0.8541 - 0.1759,
            0.8638 - 0.1543,
            0.6825 - 0.1422,
            0.6702 - 0.0361,
            0.7125 - 0.0251,
            0.6783 - 0.0267,
            0.4904 - 0.0170,
        ]
    )
)