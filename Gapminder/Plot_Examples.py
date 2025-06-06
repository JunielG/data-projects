import matplotlib.pyplot as plt
import numpy as np

# Example 1: Basic vertical bar chart with plt.bar()
plt.figure(figsize=(10, 6))

# Data
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [23, 45, 56, 78, 32]

plt.subplot(2, 2, 1)
plt.bar(categories, values)
plt.title('Basic Vertical Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Example 2: Customized vertical bar chart
plt.subplot(2, 2, 2)
colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F']
plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.title('Customized Vertical Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(rotation=45)

# Example 3: Basic horizontal bar chart with plt.barh()
plt.subplot(2, 2, 3)
plt.barh(categories, values)
plt.title('Basic Horizontal Bar Chart')
plt.xlabel('Values')
plt.ylabel('Categories')

# Example 4: Grouped bar chart
plt.subplot(2, 2, 4)
x = np.arange(len(categories))
width = 0.35
data1 = [20, 34, 30, 35, 27]
data2 = [25, 32, 34, 20, 25]

plt.bar(x - width/2, data1, width, label='Group 1')
plt.bar(x + width/2, data2, width, label='Group 2')
plt.title('Grouped Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(x, categories, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Example 5: Stacked bar chart
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
data1 = [20, 34, 30, 35, 27]
data2 = [25, 32, 34, 20, 25]
data3 = [15, 10, 12, 8, 16]

plt.bar(categories, data1, label='Layer 1')
plt.bar(categories, data2, bottom=data1, label='Layer 2')
plt.bar(categories, data3, bottom=np.array(data1) + np.array(data2), label='Layer 3')

plt.title('Stacked Vertical Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)

# Example 6: Horizontal stacked bar chart
plt.subplot(2, 1, 2)

plt.barh(categories, data1, label='Layer 1')
plt.barh(categories, data2, left=data1, label='Layer 2')
plt.barh(categories, data3, left=np.array(data1) + np.array(data2), label='Layer 3')

plt.title('Stacked Horizontal Bar Chart')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.legend()

plt.tight_layout()
plt.show()

# Example 7: Bar chart with error bars
plt.figure(figsize=(10, 6))

values = [23, 45, 56, 78, 32]
errors = [3, 5, 6, 4, 3]

plt.subplot(1, 2, 1)
plt.bar(categories, values, yerr=errors, capsize=5, 
        color='skyblue', ecolor='black', alpha=0.7)
plt.title('Vertical Bar Chart with Error Bars')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(rotation=45)

# Example 8: Horizontal bar chart with error bars
plt.subplot(1, 2, 2)
plt.barh(categories, values, xerr=errors, capsize=5,
         color='lightgreen', ecolor='black', alpha=0.7)
plt.title('Horizontal Bar Chart with Error Bars')
plt.xlabel('Values')
plt.ylabel('Categories')

plt.tight_layout()
plt.show()