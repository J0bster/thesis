import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tree_experiment_results.csv')

df = df.sort_values(['func', 'bitsize', 'depth', 'trial_tree'])

grouped = df.groupby(['func', 'bitsize', 'depth'])['avg_time'].mean().reset_index()

plt.figure(figsize=(10, 6))

from itertools import cycle
markers = cycle(['o', 's', '^', 'D', 'v', 'x', '*'])
linestyles = cycle(['-', '--', '-.', ':'])

for (func, bitsize) in grouped[['func', 'bitsize']].drop_duplicates().itertuples(index=False):
    sub = grouped[(grouped['func'] == func) & (grouped['bitsize'] == bitsize)]
    plt.plot(
        sub['depth'],
        sub['avg_time'],
        marker=next(markers),
        linestyle=next(linestyles),
        label=f"{func} (bitsize={bitsize})"
    )

plt.title("Average Evaluation Time vs Tree Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Average Time (seconds)")
plt.legend(title="Function & Bitsize")
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()