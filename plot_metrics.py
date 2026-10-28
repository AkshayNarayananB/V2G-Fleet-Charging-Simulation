import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def load_and_clean(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"Error: Log file '{csv_path}' not found.")
        return None
    
    df = pd.read_csv(csv_path)
    if df.columns[-1].startswith('Unnamed'):
        df = df.dropna(axis=1, how='all')
    return df

def plot_comparative_metrics(baseline_csv: str, ppo_csv: str):
    df_base = load_and_clean(baseline_csv)
    df_ppo = load_and_clean(ppo_csv)
    
    if df_base is None or df_ppo is None:
        return

    total_steps = min(len(df_base), len(df_ppo))
    df_base = df_base.iloc[:total_steps].copy()
    df_ppo = df_ppo.iloc[:total_steps].copy()

    # --- Match True Simulation Clock ---
    filename = os.path.basename(ppo_csv)
    match = re.search(r'(\d{4}-\d{2})', filename)
    start_date = f"{match.group(1)}-01 00:00:00" if match else "2026-01-01 00:00:00"
    
    x_axis = pd.date_range(start=start_date, periods=total_steps, freq="h")

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # =========================================================================
    # Plot 1: Operational Deliverables (Cumulative Profit / Fares)
    # =========================================================================
    if 'profit' in df_base.columns and 'profit' in df_ppo.columns:
        # FIX: Apply cumulative sum to show total profit growth over time
        axs[0].plot(x_axis, df_base['profit'].cumsum(), color='crimson', linestyle='--', linewidth=2, label='80/20 Baseline (Idle)')
        axs[0].plot(x_axis, df_ppo['profit'].cumsum(), color='green', linewidth=2, label='PPO (First Iteration)')
        axs[0].set_ylabel('Total Operational Value ($)')
    else:
        axs[0].text(0.5, 0.5, 'Operational Data Missing', horizontalalignment='center')
        
    axs[0].set_title('Operational Goal Telemetry (Cumulative Fares)', fontsize=14)
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # =========================================================================
    # Plot 2: Dynamic Synthetic Load & Peak Shaving
    # =========================================================================
    t_hours = np.array([t.hour + (t.minute / 60.0) for t in x_axis])
    baseline_load = 100.0 + 50.0 * np.sin((2.0 * np.pi / 24.0) * (t_hours - 12.0))
    
    p_col_base = 'total_power' if 'total_power' in df_base.columns else 'total_grid_power'
    p_col_ppo = 'total_power' if 'total_power' in df_ppo.columns else 'total_grid_power'
    
    net_load_base = baseline_load + df_base[p_col_base].fillna(0) if p_col_base in df_base.columns else baseline_load
    net_load_ppo = baseline_load + df_ppo[p_col_ppo].fillna(0) if p_col_ppo in df_ppo.columns else baseline_load

    # FIX: Apply a 12-hour rolling average to smooth the noisy power spikes
    smoothed_load_base = pd.Series(net_load_base).rolling(window=12, min_periods=1).mean()
    smoothed_load_ppo = pd.Series(net_load_ppo).rolling(window=12, min_periods=1).mean()

    axs[1].plot(x_axis, baseline_load, color='gray', linestyle=':', label='Dynamic Baseline Load (No V2G)', alpha=0.8, linewidth=2)
    axs[1].plot(x_axis, smoothed_load_base, color='crimson', linestyle='--', label='Net Load (80/20 Baseline)', linewidth=1.5, alpha=0.8)
    axs[1].plot(x_axis, smoothed_load_ppo, color='orange', label='Net Load (PPO - 12h Smoothed)', linewidth=2)
    axs[1].axhline(y=300.0, color='red', linestyle='-', alpha=0.9, linewidth=2, label='Peak Limit Threshold (300 kW)')
    
    axs[1].set_title('Building Energy Peak Shaving (Smoothed Demand)', fontsize=14)
    axs[1].set_ylabel('Power (kW)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # =========================================================================
    # Plot 3: Fleet Battery Degradation (SoH)
    # =========================================================================
    soh_cols_base = [col for col in df_base.columns if 'soh' in col.lower()]
    soh_cols_ppo = [col for col in df_ppo.columns if 'soh' in col.lower()]
    
    if soh_cols_base and soh_cols_ppo:
        avg_soh_base = df_base[soh_cols_base].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        avg_soh_ppo = df_ppo[soh_cols_ppo].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        
        axs[2].plot(x_axis, avg_soh_base, color='crimson', linestyle='--', linewidth=2, label='80/20 Baseline SoH')
        axs[2].plot(x_axis, avg_soh_ppo, color='blue', linewidth=2, label='PPO (First Iteration) SoH')
        
        axs[2].set_title('Fleet Average Battery State of Health (SoH)', fontsize=14)
        axs[2].set_ylabel('SoH (%)')
        axs[2].legend(loc='lower left')
        axs[2].grid(True, linestyle='--', alpha=0.6)

    # --- FIX: X-Axis Formatting to remove the solid black bar ---
    # Automatically select a readable number of ticks instead of plotting every single step
    locator = mdates.AutoDateLocator(maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    
    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(x_axis[0], x_axis[-1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparative_metrics(
        baseline_csv="logs/baseline_8020.csv", 
        ppo_csv="logs/baseline_ppo.csv"
    )