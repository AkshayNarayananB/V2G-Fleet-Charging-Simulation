import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def plot_fleet_states(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    
    status_cols = [col for col in df.columns if col.startswith('status')]
    state_df = df[status_cols]
    
    state_mapping = {
        'IDLE': 0, 'INSERVICE': 1, 'TOPICKUP': 2, 
        'TOCHARGE': 3, 'CHARGING': 4, 'RECOVERY': 5
    }
    colors = ['#d3d3d3', '#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
    
    numeric_states = state_df.applymap(lambda x: state_mapping.get(str(x).strip(), 0))
    state_matrix = numeric_states.T.values
    
    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = ListedColormap(colors)
    
    cax = ax.pcolormesh(state_matrix, cmap=cmap, vmin=0, vmax=len(colors)-1)
    
    ax.set_title(f'Fleet State Timeline ({csv_path})', fontsize=16)
    ax.set_xlabel('Simulation Step (Time)', fontsize=12)
    ax.set_ylabel('Vehicle ID', fontsize=12)
    
    legend_patches = [mpatches.Patch(color=colors[i], label=state_name) for state_name, i in state_mapping.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

# Inside plot_states.py, change the bottom lines to run on both:
plot_fleet_states("logs/baseline_8020.csv", "logs/state_chart_8020.png")
plot_fleet_states("logs/baseline_ppo.csv", "logs/state_chart_ppo.png")