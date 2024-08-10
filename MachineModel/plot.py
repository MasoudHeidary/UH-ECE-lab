import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


manipulate_percentage = [0, 1, 2, 3, 4]

# network Cipher10 [epoch:100], using original delta 
darknetMani_delta1 = [0.8358, 0.7917, 0.7393, 0.6705]
squeezenetMani_delta1 = [0.8397, 0.7411, 0.5843, 0.4164]
resnet18Mani_delta1 = [0.8666, 0.8521, 0.8258, 0.7981]
vgg11Mani_delta1 = [0.8614, 0.8388, 0.8121, 0.7828]
vgg19Mani_delta1 = [0.8891, 0.8766, 0.8578, 0.833]

# plt.plot(manipulate_percentage, darknetMani_delta1, linewidth=2, label="darknet d[1]")
# plt.plot(manipulate_percentage, squeezenetMani_delta1, linewidth=2, label="squeeze d[1]")
# plt.plot(manipulate_percentage, resnet18Mani_delta1, linewidth=3, label="resnet d[1]")
# plt.plot(manipulate_percentage, vgg11Mani_delta1, linewidth=3, label="vgg11 d[1]")
# plt.plot(manipulate_percentage, vgg19Mani_delta1, linewidth=2, label="vgg19 d[1]")

# plt.text(manipulate_percentage[-1], darknetMani_delta1[-1], "darknet d[1]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], squeezenetMani_delta1[-1], "squeeze d[1]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], resnet18Mani_delta1[-1], "resnet d[1]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg11Mani_delta1[-1], "vgg11 d[1]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg19Mani_delta1[-1], "vgg19 d[1]", fontsize=14, fontweight='bold', ha='right', va='bottom') 

# network Cipher10 [epoch:100], using x2 delta
darknetMani_delta2 = [0.8358, 0.5617, 0.2006, 0.1135, 0.1034]
squeezenetMani_delta2 = [0.8397, 0.2692, 0.1251, 0.1053, 0.1010]
resnet18Mani_delta2 = [0.8666, 0.7486, 0.5593, 0.3686, 0.2440]
vgg11Mani_delta2 = [0.8614, 0.7226, 0.4991, 0.3071, 0.2055]
vgg19Mani_delta2 = [0.8891, 0.776, 0.4828, 0.1949, 0.1184]

if True:
    plt.figure(figsize=(13, 10))

    plt.plot(manipulate_percentage, darknetMani_delta2, linewidth=5, label="Darknet")
    plt.plot(manipulate_percentage, squeezenetMani_delta2, linewidth=5, label="Squeeze")
    plt.plot(manipulate_percentage, resnet18Mani_delta2, linewidth=5, label="ResNet")
    plt.plot(manipulate_percentage, vgg11Mani_delta2, linewidth=5, label="VGG11")
    plt.plot(manipulate_percentage, vgg19Mani_delta2, linewidth=5, label="VGG19")


    plt.xlabel('Number of Errors (%)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28, fontweight='bold')

    plt.ylabel('Network Accuracy (CIFAR10)', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    # plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    plt.grid(True)
    plt.legend(fontsize=28)
    plt.show()

# plt.text(manipulate_percentage[-1], darknetMani_delta2[-1], "darknet d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], squeezenetMani_delta2[-1], "squeeze d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], resnet18Mani_delta2[-1], "resnet d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg11Mani_delta2[-1], "vgg11 d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg19Mani_delta2[-1], "vgg19 d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 






# SVHN, using regular (x1) delta
manipulate_percentage = [0, 1, 2, 3, 4]
darknet_SVHN = [0.8575, 0.6707, 0.3628, 0.1934, 0.1696]
squeezenet_SVHN = [0.8726, 0.4841, 0.2041, 0.1520, 0.1504]
resnet18_SVHN = [0.8452, 0.7469, 0.5632, 0.3660, 0.2454]
vgg11_SVHN = [0.8541, 0.6541, 0.3893, 0.2393, 0.1759]
vgg19_SVHN = [0.8638, 0.7722, 0.5138, 0.2406, 0.1543]

if True:
    plt.figure(figsize=(13, 10))

    plt.plot(manipulate_percentage, darknet_SVHN, linewidth=5, label="Darknet")
    plt.plot(manipulate_percentage, squeezenet_SVHN, linewidth=5, label="Squeeze")
    plt.plot(manipulate_percentage, resnet18_SVHN, linewidth=5, label="ResNet")
    plt.plot(manipulate_percentage, vgg11_SVHN, linewidth=5, label="VGG11")
    plt.plot(manipulate_percentage, vgg19_SVHN, linewidth=5, label="VGG19")


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