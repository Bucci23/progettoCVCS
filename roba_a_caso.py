import matplotlib.pyplot as plt
import json

# Sample dictionary

with open(f"factors.json", "r") as read_file:
    data = json.load(read_file)
# Convert dictionary to lists
labels = [1.25, 1.5, 1.75, 2, 2.25, 2.5]
values = [0.8415993665973733, 0.8444767708860864, 0.8461985197712157, 0.8446379135218306, 0.8463400377248615,
          0.8459932596715956]

# Set variances to zero
variances = [0] * len(values)

# Plot the data
plt.errorbar(labels, values, yerr=variances, fmt='o', capsize=4)

# Add title and axis labels
plt.title('Best IoUs for each factor')
plt.xlabel('THH/THL')
plt.ylabel('IOUs')
plt.ylim([0.83, 0.855])
# Show the plot
plt.savefig("definitive.png", dpi=300)
plt.show()

