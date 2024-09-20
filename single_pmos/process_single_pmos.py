from tool.log import Log
import re
import matplotlib.pyplot as plt

def filter_data(data):
    pattern = r"([\d.]+)[a-zA-Z]?"
    matches = re.findall(pattern, data)
    
    # Convert the numeric parts to floats
    filtered_matches = [float(match) for match in matches]

    return filtered_matches[0]








if __name__ == "__main__":
    log = Log("terminal-log.txt", terminal=True)

    if True:
        plot_pb = []
        plot_current_u = [] 

        for body_voltage in range(800, 5000, 10):
            pb = body_voltage / 1000
            
            log_name = f"./raw_data_current_on_vb/pmos-pb-{pb:.2f}.txt"
            log_file = open(log_name)
            current_u = filter_data(log_file.read())
            
            plot_pb.append(body_voltage)
            plot_current_u.append(current_u)

            log.println(
                f"pb:{pb:.2f}\tI:{current_u}"
            )

        
        # plot
        if True:
            plt.figure(figsize=(13, 10))

            plt.plot(plot_pb, plot_current_u, label=".", linewidth=5)

            plt.xlabel('body voltage', fontsize=28, fontweight='bold')
            plt.xticks(fontsize=28, fontweight='bold')

            plt.ylabel('current', fontsize=28, fontweight='bold')
            plt.yticks(fontsize=28, fontweight='bold')

            plt.legend(fontsize=28)
            plt.grid()
            plt.show()