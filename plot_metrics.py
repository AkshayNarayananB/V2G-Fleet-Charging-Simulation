import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plot_evaluation_metrics(csv_path="logs/combined_eval.csv"):
    df = pd.read_csv(csv_path)
    
    # Using your exact confirmed 1-hour step size starting from your dataset window
    x_axis = pd.date_range(start="2026-01-01 00:00:00", periods=len(df), freq="h")
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # --- 1. Fleet Manager Profit ---
    axs[0].plot(x_axis, df['profit'].cumsum(), color='green', linewidth=2)
    axs[0].set_title('Cumulative Fleet Profit / Reward Over Time')
    axs[0].set_ylabel('Profit ($ / Reward)')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- 2. Building Energy Peak Shaving ---
    baseline_load = 250.0 
    net_building_load = baseline_load + df['total_power']
    
    axs[1].plot(x_axis, [baseline_load] * len(df), 'r--', label='Baseline Load (No V2G)')
    axs[1].plot(x_axis, net_building_load, color='orange', label='Net Load (With V2G)', linewidth=1.5)
    axs[1].axhline(y=300, color='red', linestyle='-', alpha=0.5, label='Peak Limit Penalty Threshold (300 kW)')
    axs[1].set_title('Building Load & Peak Shaving (V2G Impact)')
    axs[1].set_ylabel('Power (kW)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- 3. Battery Degradation (State of Health) ---
    soh_cols = [col for col in df.columns if col.startswith('soh')]
    avg_soh = df[soh_cols].mean(axis=1)
    axs[2].plot(x_axis, avg_soh, color='blue', linewidth=2, label='PPO Managed V2G SoH')
    
    unmanaged_soh = 1.0 - (np.arange(len(df)) * (1.0 - avg_soh.iloc[-1]) * 1.5 / len(df))
    axs[2].plot(x_axis, unmanaged_soh, color='red', linestyle='--', label='Unmanaged Baseline (Est.)')
    
    axs[2].set_title('Fleet Average Battery Degradation (State of Health)')
    axs[2].set_ylabel('SoH (%)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    # --- X-Axis Formatting for Multi-Year Timeline ---
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)) # Clean labels every 6 months
    
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_evaluation_metrics("logs/nyc_eval.csv")