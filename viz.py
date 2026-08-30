import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dyp_metrics_and_states(csv_file, peak_threshold_kw=500.0, peak_penalty=0.25, deg_cost=0.05, delta_t=1.0, battery_capacity=50.0):
    # 1. Load Data
    df = pd.read_csv(csv_file)
    
    # 2. Derive Power mathematically from SoC diff (bypasses the missing power_kw key)
    # Sort to ensure time steps are sequential for each vehicle
    df = df.sort_values(by=['Vehicle_ID', 'Time_Step'])
    
    # Difference in SoC * Battery Capacity / Time = kW Power
    df['SoC_Diff'] = df.groupby('Vehicle_ID')['SoC'].diff().fillna(0)
    df['Calculated_Power_kW'] = (df['SoC_Diff'] * battery_capacity) / delta_t
    
    # 3. Global Ride Metrics 
    # Since it's a cumulative counter in your logger, we just take the max value
    completed_rides = df['Completed_Rides_Global'].max()
    skipped_rides = df['Skipped_Rides_Global'].max()
    
    # 4. Financial & Battery Metrics
    # Discharging Income (Power < 0 means V2G)
    df['Discharge_kW'] = df['Calculated_Power_kW'].apply(lambda x: abs(x) if x < 0 else 0)
    df['Discharge_Income'] = df['Discharge_kW'] * delta_t * df['Grid_Price']
    total_discharge_income = df['Discharge_Income'].sum()
    
    # Degradation Cost
    df['Throughput_kWh'] = df['Calculated_Power_kW'].abs() * delta_t
    total_degradation_cost = df['Throughput_kWh'].sum() * deg_cost
    
    # Peak Shaving
    time_grouped = df.groupby('Time_Step').agg({
        'Building_Load_kW': 'first',
        'Calculated_Power_kW': 'sum'  # Net EV fleet power
    }).reset_index()
    
    time_grouped['Net_Load'] = time_grouped['Building_Load_kW'] + time_grouped['Calculated_Power_kW']
    
    baseline_above_thresh = np.maximum(0, time_grouped['Building_Load_kW'] - peak_threshold_kw)
    net_above_thresh = np.maximum(0, time_grouped['Net_Load'] - peak_threshold_kw)
    time_grouped['Peak_Shaved_kWh'] = (baseline_above_thresh - net_above_thresh).clip(lower=0) * delta_t
    
    total_peak_shaving_income = time_grouped['Peak_Shaved_kWh'].sum() * peak_penalty

    # Print Summary Metrics
    print(f"--- DYP Fleet Simulation Results ---")
    print(f"Completed Rides:          {int(completed_rides) if pd.notna(completed_rides) else 0}")
    print(f"Skipped Rides:            {int(skipped_rides) if pd.notna(skipped_rides) else 0}")
    print(f"Overall Discharge Income: ${total_discharge_income:.2f}")
    print(f"Pure Peak Shaving Income: ${total_peak_shaving_income:.2f}")
    print(f"Battery Degradation Cost: ${total_degradation_cost:.2f}")
    print(f"Net Financial Yield:      ${(total_discharge_income + total_peak_shaving_income) - total_degradation_cost:.2f}\n")

    # 5. Combined Fleet State Matrix Heatmap
    state_matrix = df.groupby(['State', 'Time_Step']).size().unstack(fill_value=0)
    total_vehicles = df['Vehicle_ID'].nunique()
    
    # Format annotations
    annot = np.empty(state_matrix.shape, dtype=object)
    for r in range(state_matrix.shape[0]):
        for c in range(state_matrix.shape[1]):
            val = state_matrix.iloc[r, c]
            if val == 0:
                annot[r, c] = ""
            else:
                annot[r, c] = f"{int(val)}\n{(val/total_vehicles)*100:.0f}%"

    fig, ax = plt.subplots(figsize=(20, 5))
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(state_matrix, ax=ax, cmap=cmap, annot=annot, fmt="", linewidths=0.5)
    
    ax.set_title("Combined Fleet State Matrix", fontsize=14, pad=15)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Vehicle State", fontsize=12)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('combined_state_matrix.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    import sys
    csv_input = sys.argv[1] if len(sys.argv) > 1 else 'telemetry.csv'
    analyze_dyp_metrics_and_states(csv_input)
