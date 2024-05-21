import matplotlib.pyplot as plt
import lib.NBTI_formula as NBTI


categories = [f"FA-{i}" for i in range(12)]
normal_alpha = [160, 160, 160, 192, 144, 160, 164, 192, 136, 148, 164, 192]
modified_alpha = [192, 208, 160, 192, 192, 232, 164, 192, 192, 196, 198, 202]


# Define the width of the bars
bar_width = 0.35

# Set the position of the bars on the x-axis
r1 = range(len(categories))
r2 = [x + bar_width for x in r1]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(r1, normal_alpha, color='skyblue', width=bar_width, label='Before')
plt.bar(r2, modified_alpha, color='orange', width=bar_width, label='After')

# Add xticks on the middle of the group bars
plt.xlabel('Categories', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(categories))], categories)

# Add a legend
plt.legend()

# Show plot
plt.title('Before and After Comparison')
plt.show()

