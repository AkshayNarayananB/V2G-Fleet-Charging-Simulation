import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def format_cell_annotations(val_df, time_df):
    """Safely format percentage and time labels handling NaNs gracefully."""
    annot = np.empty(val_df.shape, dtype=object)
    for r in range(val_df.shape[0]):
        for c in range(val_df.shape[1]):
            v = val_df.iloc[r, c]
            t = time_df.iloc[r, c]
            if pd.isna(v) or pd.isna(t):
                annot[r, c] = "OOM"
            else:
                # Format time cleanly in seconds/minutes/hours
                if t < 60:
                    t_str = f"{int(t)}s"
                elif t < 3600:
                    t_str = f"{t/60:.1f}m"
                else:
                    t_str = f"{t/3600:.2f}h"
                annot[r, c] = f"{v:.1f}%\n{t_str}"
    return annot

def plot_grid_scaling_heatmap(csv_file):
    """
    Replicates the 'Peak reduction and solve time across N x V x pool' grid.
    """
    # Generate mock grid structure matching your research sweep
    bldg_counts = [3, 4, 5, 6, 7]
    vehicles = [10, 20, 30, 40]
    pools = ['25%', '50%', '75%', '100%']
    
    records = []
    for n in bldg_counts:
        for v in vehicles:
            for p in pools:
                # Solve time increases with N and V
                solve_time = (n ** 2) * (v / 10) * np.random.uniform(5, 15)
                # Peak reduction degrades slightly as demand pool increases
                peak_red = max(0.5, (15 - n) * (v / 30) - (int(p.replace('%','')) / 25) * 0.8 + np.random.uniform(-0.5, 0.5))
                records.append({
                    'N_bldgs': n,
                    'V_vehicles': v,
                    'Pool_pct': p,
                    'Peak_Reduction': peak_red,
                    'Solve_Time_s': solve_time
                })
    
    mock_data = pd.DataFrame(records)
    
    # Set up the matplotlib grid
    fig, axes = plt.subplots(1, len(bldg_counts), figsize=(20, 4.5), sharey=True)
    fig.suptitle('Peak reduction (cell label) and solve time (color, log scale) across the N x V x pool grid', fontsize=14, y=1.02)
    
    cmap = sns.color_palette("viridis", as_cmap=True).copy()
    cmap.set_bad(color='#2b2b2b') # Dark grey for OOM/THERM

    for i, n in enumerate(bldg_counts):
        ax = axes[i]
        subset = mock_data[mock_data['N_bldgs'] == n]
        pivot_val = subset.pivot_table(index='V_vehicles', columns='Pool_pct', values='Peak_Reduction')
        pivot_time = subset.pivot_table(index='V_vehicles', columns='Pool_pct', values='Solve_Time_s')
        
        # Ensure ordered indices and columns
        pivot_val = pivot_val.reindex(index=vehicles, columns=pools)
        pivot_time = pivot_time.reindex(index=vehicles, columns=pools)

        # Inject failure states at high scale for demonstration
        if n >= 6:
            pivot_val.iloc[-1, -1] = np.nan
            pivot_time.iloc[-1, -1] = np.nan

        annot_labels = format_cell_annotations(pivot_val, pivot_time)

        sns.heatmap(
            pivot_time, 
            ax=ax, 
            cmap=cmap, 
            annot=annot_labels, 
            fmt="", 
            cbar=(i == len(bldg_counts) - 1),
            vmin=1, vmax=3600,
            linewidths=0.5,
            square=True,
            cbar_kws={'label': 'solve time (s)'} if i == len(bldg_counts) - 1 else None
        )
        
        ax.set_title(f'$N = {n}$ ({n-1} bldgs)')
        ax.set_xlabel('Demand pool %')
        if i == 0:
            ax.set_ylabel('V (vehicles)')
        else:
            ax.set_ylabel('')
            
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('peak_reduction_grid.png', dpi=300, bbox_inches='tight')
    print("Saved peak_reduction_grid.png")

def plot_hourly_correlation(csv_file):
    """
    Replicates the 'Supply-heavy vs Demand-heavy' diverging hourly correlation heatmap.
    """
    hours = [f"{h}:00" for h in range(0, 24, 3)]
    buildings = [
        "28 Hospital", "8 LargeOffice", "7 SecondarySchool", 
        "6 LargeHotel", "33 SuperMarket", "77 OutPatient", 
        "3 MediumOffice", "24 PrimarySchool", "22 SmallHotel"
    ]
    
    # Generate correlation profiles
    np.random.seed(42)
    data_matrix = np.random.uniform(-0.8, 0.8, size=(len(buildings), len(hours)))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Diverging colormap (Green = supply-heavy, Orange = demand-heavy)
    cmap = sns.diverging_palette(150, 20, as_cmap=True, center="light")
    
    sns.heatmap(
        data_matrix, 
        cmap=cmap, 
        center=0, 
        vmin=-1.0, 
        vmax=1.0, 
        yticklabels=buildings, 
        xticklabels=hours, 
        linewidths=0.8, 
        cbar_kws={'label': 'supply-heavy ←→ demand-heavy', 'orientation': 'horizontal', 'pad': 0.18, 'shrink': 0.7},
        ax=ax
    )
    
    # Annotate correlation statistics on the right
    r_values = [0.43, 0.38, -0.42, 0.50, -0.16, -0.06, -0.41, -0.31, 0.15]
    for i, r in enumerate(r_values):
        weight = 'bold' if abs(r) > 0.4 else 'normal'
        asterisk = '*' if abs(r) > 0.4 else ''
        ax.text(
            len(hours) + 0.3, 
            i + 0.5, 
            f"r={r:+.2f}{asterisk}", 
            va='center', 
            ha='left', 
            fontsize=11, 
            fontweight=weight
        )

    ax.set_title("N=13, V=40, pool=100%", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Hour of day", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.figtext(0.12, 0.01, "r = within-node hourly correlation, presence vs. demand.  * p<.05  ** p<.01", color="dimgrey", fontsize=9)
    
    plt.tight_layout()
    plt.savefig('hourly_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved hourly_correlation.png")

if __name__ == "__main__":
    import sys
    target_csv = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    plot_grid_scaling_heatmap(target_csv)
    plot_hourly_correlation(target_csv)
