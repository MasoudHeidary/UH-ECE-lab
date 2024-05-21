import re

pattern = r"([0-9.]+)([a-zA-Z]+)"

vb_list = [i/100 for i in range(80, 500, 2)]
vth_list = []

for vb in vb_list:
    file_name = f"./raw_data/{vb}.txt.txt"

    file = open(file_name)
    file.readline()
    file.readline()
    file.readline()

    num = []
    for line in file.readlines():
        match = re.findall(pattern, line)
        num += [match]
    
    for rev in reversed(num):
        if(rev[1][1] == 'u'):
            if(float(rev[1][0]) > 10):
                vth = round(float(rev[0][0]) / 1000 - 0.8, 3)
                vth_list += [vth]
                # print(vb, "\t===\t", vth)
                print(f"{vth*-1}, ")
                break

    file.close()



# write the data to file
if False:
    file_out = open("pmos_vth_vbody.txt", "w")
    
    for vb_index, vb in enumerate(vb_list):
        vdef = round(vth_list[vb_index] - vth_list[0], 3)
        file_out.write(f"{vb}\t\t{vth_list[vb_index]}\t\t{vdef}\n")

    file_out.close()

        

# plot
if False:
    import matplotlib.pyplot as plt 

    x = vb_list.copy()
    y = vth_list.copy()
    plt.plot(x, y)
    plt.show()