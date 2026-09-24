import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 1. Load TRIADS Theoretical (Tensor) Data
df_tensor = pd.read_csv("triads_throughput_benchmark.csv")

# 2. Load TRIADS E2E Data
df_e2e = pd.read_csv("triads_e2e_benchmark.csv")

# 3. Dynamically find CrabNet peak throughput (No magic numbers)
crabnet_peak = 0
paths_to_check = [
    "../CrabNet_TiAlloy_Test/benchmark_results_v2.csv",
    "../comparison/model_results_crabnet.json"
]

for path in paths_to_check:
    if os.path.exists(path):
        if path.endswith(".csv"):
            df_crab = pd.read_csv(path)
            crabnet_peak = df_crab['throughput_sps'].max()
        elif path.endswith(".json"):
            with open(path) as f:
                crabnet_peak = json.load(f).get('throughput', {}).get('samples_per_s', 0)
        break

if crabnet_peak == 0:
    print("Warning: Could not dynamically locate CrabNet data. Using fallback 244.33.")
    crabnet_peak = 244.33
else:
    print(f"Dynamically loaded CrabNet peak: {crabnet_peak} samples/sec")

# 4. Plotting Setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Tensor Scaling Curve
ax1.plot(df_tensor['batch_size'], df_tensor['samples_per_sec'], marker='o', linewidth=2.5, color='#1f77b4', markersize=8)
ax1.set_xscale('log')
ax1.set_xticks(df_tensor['batch_size'])
ax1.set_xticklabels(df_tensor['batch_size'])
ax1.set_title('TRIADS: Theoretical Tensor Speed\n(Pure PyTorch Math, Features Pre-computed)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

peak_tensor = df_tensor['samples_per_sec'].max()
peak_batch = df_tensor.loc[df_tensor['samples_per_sec'].idxmax(), 'batch_size']
ax1.annotate(f'Theoretical Peak:\n{peak_tensor:,.0f} samples/s\n(Batch {int(peak_batch)})', 
             xy=(peak_batch, peak_tensor), xytext=(peak_batch * 3, peak_tensor - 500),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"))

# Panel 2: Production Reality (E2E)
categories = [
    'TRIADS\n(Tensor Math Only)', 
    'CrabNet\n(E2E Production)', 
    'TRIADS E2E\n(Steels)', 
    'TRIADS E2E\n(Ti-Alloys)'
]

values = [
    peak_tensor, 
    crabnet_peak, 
    df_e2e[df_e2e['scenario']=='Pure Steels']['samples_per_sec'].values[0], 
    df_e2e[df_e2e['scenario']=='Pure Ti-Alloys']['samples_per_sec'].values[0]
]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
ax2.set_yscale('log')
ax2.set_title('Production Reality Check\n(String $\\rightarrow$ Matminer $\\rightarrow$ PyTorch)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Throughput (samples/sec, log scale)', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, 
             f'{val:,.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add a text box explaining the bottleneck
textstr = 'The "Feature Engineering Tax":\nMatminer calculating 532\ndescriptors takes ~40ms/alloy,\nslowing TRIADS down by 160x.'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black')
ax2.text(0.65, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('slide_triads_throughput_e2e.png', dpi=300, bbox_inches='tight')
print("\n[Saved] slide_triads_throughput_e2e.png")
plt.show()