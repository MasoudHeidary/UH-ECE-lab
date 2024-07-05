import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


manipulate_percentage = [0, 1, 2, 3]

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
darknetMani_delta2 = [0.8358, 0.5617, 0.2006, 0.1135,]
squeezenetMani_delta2 = [0.8397, 0.2692, 0.1251, 0.1053,]
resnet18Mani_delta2 = [0.8666, 0.7486, 0.5593, 0.3686,]
vgg11Mani_delta2 = [0.8614, 0.7226, 0.4991, 0.3071,]
vgg19Mani_delta2 = [0.8891, 0.776, 0.4828, 0.1949,]

if False:
    plt.plot(manipulate_percentage, darknetMani_delta2, linewidth=3, label="darknet")
    plt.plot(manipulate_percentage, squeezenetMani_delta2, linewidth=3, label="squeeze")
    plt.plot(manipulate_percentage, resnet18Mani_delta2, linewidth=3, label="resnet")
    plt.plot(manipulate_percentage, vgg11Mani_delta2, linewidth=3, label="vgg11")
    plt.plot(manipulate_percentage, vgg19Mani_delta2, linewidth=3, label="vgg19")

# plt.text(manipulate_percentage[-1], darknetMani_delta2[-1], "darknet d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], squeezenetMani_delta2[-1], "squeeze d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], resnet18Mani_delta2[-1], "resnet d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg11Mani_delta2[-1], "vgg11 d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 
# plt.text(manipulate_percentage[-1], vgg19Mani_delta2[-1], "vgg19 d[2]", fontsize=14, fontweight='bold', ha='right', va='bottom') 


# SVHN, using regular (x1) delta
manipulate_percentage = [0, 1, 2, 3, 4, 5]
darknet_SVHN = [0.8575, 0.6707, 0.3628, 0.1934, 0.1696, 0.1660]
squeezenet_SVHN = [0.8726, 0.4841, 0.2041, 0.1520, 0.1504, 0.1500]
resnet18_SVHN = [0.8452, 0.7469, 0.5632, 0.3660, 0.2454, 0.1908]
vgg11_SVHN = [0.8541, 0.6541, 0.3893, 0.2393, 0.1759, 0.1535]
vgg19_SVHN = [0.8638, 0.7722, 0.5138, 0.2406, 0.1543, 0.1423]

if True:
    plt.plot(manipulate_percentage, darknet_SVHN, linewidth=3, label="darknet")
    plt.plot(manipulate_percentage, squeezenet_SVHN, linewidth=3, label="squeeze")
    plt.plot(manipulate_percentage, resnet18_SVHN, linewidth=3, label="resnet")
    plt.plot(manipulate_percentage, vgg11_SVHN, linewidth=3, label="vgg11")
    plt.plot(manipulate_percentage, vgg19_SVHN, linewidth=3, label="vgg19")


plt.xlabel('error rate percentage', fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')

plt.ylabel('network accuracy', fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# plt.title('Title', fontsize=16, fontweight='bold')
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
plt.grid(True)
plt.legend(fontsize=14)
plt.show()
