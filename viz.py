import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot(csv_file, fare_per_ride=20.0, battery_cost_per_kwh=135.0, peak_limit_kw=500.0):
    print(f"Loading and analyzing telemetry from: {csv_file}...")
    df = pd.read_csv(csv_file)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # 1. Group by Time_Step (1 row per hour) so global step metrics aren't multiplied by 50 vehicles
    step_df = df.groupby('Time_Step', as_index=False).first().sort_values('Time_Step')

    # 2. Calculate True Cumulative Ride & V2B Metrics
    completed_rides = int(step_df['Completed_Rides_Global'].sum())

    if 'Skipped_Rides_Global' in step_df.columns:
        skipped_vals = step_df['Skipped_Rides_Global'].values
        # Auto-detect if skipped_rides was logged as a cumulative counter or per-step counter
        if len(skipped_vals) > 1 and np.all(np.diff(skipped_vals) >= 0) and skipped_vals[-1] > skipped_vals[0] * 5:
            skipped_rides = int(skipped_vals[-1])
        else:
            skipped_rides = int(step_df['Skipped_Rides_Global'].sum())
    else:
        skipped_rides = 0

    taxi_profit = completed_rides * fare_per_ride
    v2b_income = float(step_df['V2B_Revenue_Global'].sum()) if 'V2B_Revenue_Global' in step_df.columns else 0.0

    # 3. Calculate True Battery Degradation Cost from SoH (Initial - Final SoH across the 50-car fleet)
    # BYD E6 initial battery capacity = 60.0 kWh (or 50.0 kWh depending on model config)
    INITIAL_CAP_KWH = 60.0
    veh_soh = df.sort_values('Time_Step').groupby('Vehicle_ID')['SoH'].agg(['first', 'last'])
    total_soh_fraction_lost = (veh_soh['first'] - veh_soh['last']).clip(lower=0.0).sum()
    degradation_cost = float(total_soh_fraction_lost * INITIAL_CAP_KWH * battery_cost_per_kwh)

    net_yield = taxi_profit + v2b_income - degradation_cost

    # 4. Print Console Summary
    print(f"\n--- FYP Fleet Simulation Results: {csv_file} ---")
    print(f"Completed Rides:          {completed_rides}")
    print(f"Skipped Rides:            {skipped_rides}")
    print(f"Estimated Taxi Profit:    ${taxi_profit:.2f}")
    print(f"Cumulative V2B Income:    ${v2b_income:.2f}")
    print(f"Battery Degradation Cost: ${degradation_cost:.2f}")
    print(f"Net Financial Yield:      ${net_yield:.2f}\n")

    # Determine status column name ('Status' vs 'State')
    status_col = 'Status' if 'Status' in df.columns else 'State'
    df[status_col] = df[status_col].astype(str).apply(lambda s: s.split('.')[-1])

    print("Fleet Status Breakdown (% of vehicle-hours):")
    status_pcts = df[status_col].value_counts(normalize=True) * 100
    for st, pct in status_pcts.items():
        print(f"  - {st:<12}: {pct:.2f}%")

    # =========================================================================
    # PLOT 1: Combined Fleet State Matrix Heatmap
    # =========================================================================
    state_matrix = df.groupby([status_col, 'Time_Step']).size().unstack(fill_value=0)
    
    fig1, ax1 = plt.subplots(figsize=(18, 5))
    cmap = sns.color_palette("viridis", as_cmap=True)
    # annot=False prevents unreadable black text smudging across 750+ timesteps
    sns.heatmap(
        state_matrix,
        ax=ax1,
        cmap=cmap,
        annot=False,
        cbar_kws={'label': 'Vehicle Count (out of 50)'}
    )
    ax1.set_title(f"Combined Fleet State Matrix ({base_name})", fontsize=14, pad=12)
    ax1.set_xlabel("Time Step (Hours)", fontsize=12)
    ax1.set_ylabel("Vehicle Status", fontsize=12)
    plt.yticks(rotation=0)
    
    matrix_png = f"{base_name}_state_matrix.png"
    plt.tight_layout()
    plt.savefig(matrix_png, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n[PLOT SAVED] -> {matrix_png}")

    # =========================================================================
    # PLOT 2: 4-Panel FYP Telemetry & Financial Dashboard
    # =========================================================================
    fig2, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle(f"FYP V2G Fleet Evaluation Dashboard ({base_name})", fontsize=16, fontweight='bold')

    time_steps = step_df['Time_Step'].values
    net_load = step_df['Net_Building_Load_kW'].values
    ev_power = step_df['Total_Grid_Power_kW'].values
    baseline_load = net_load - ev_power

    # Panel 1: Microgrid Load & 500kW Peak Ceiling
    ax_grid = axs[0, 0]
    ax_grid.plot(time_steps, baseline_load, label="Baseline Building Load", linestyle="--", color="gray", alpha=0.8)
    ax_grid.plot(time_steps, net_load, label="Net Building Load (With EVs)", color="#1f77b4", linewidth=1.2)
    ax_grid.axhline(peak_limit_kw, color="red", linestyle=":", linewidth=2, label=f"Grid Ceiling ({peak_limit_kw} kW)")
    ax_grid.fill_between(
        time_steps, net_load, peak_limit_kw,
        where=(net_load > peak_limit_kw), color='red', alpha=0.3, label="Peak Violation"
    )
    ax_grid.set_title("Microgrid Load & Peak Shaving Profile")
    ax_grid.set_xlabel("Time Step (Hours)")
    ax_grid.set_ylabel("Power (kW)")
    ax_grid.legend(loc="upper right")
    ax_grid.grid(True, alpha=0.2)

    # Panel 2: Financial Yield Breakdown
    ax_fin = axs[0, 1]
    labels = ['Taxi Profit', 'V2B Income', 'Degradation Cost', 'Net Yield']
    values = [taxi_profit, v2b_income, -degradation_cost, net_yield]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    bars = ax_fin.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
    ax_fin.axhline(0, color='black', linewidth=1)
    ax_fin.set_title("Financial Breakdown ($)")
    ax_fin.set_ylabel("USD ($)")
    ax_fin.grid(True, axis='y', alpha=0.2)
    for bar in bars:
        yval = bar.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        ax_fin.text(bar.get_x() + bar.get_width()/2.0, yval, f"${yval:,.0f}", va=va, ha='center', fontweight='bold')

    # Panel 3: Fleet Status Distribution (Pie Chart)
    ax_pie = axs[1, 0]
    valid_status = status_pcts[status_pcts > 0.5]
    # ax_pie.pie(valid_status.values, labels=valid_status.index, autopct='%1.1f%%', startangle=140, cmap=plt.cm.Set2)
    pie_colors = plt.cm.Set2(np.linspace(0, 1, len(valid_status)))
    ax_pie.pie(valid_status.values, labels=valid_status.index, autopct='%1.1f%%', startangle=140, colors=pie_colors)
    ax_pie.set_title("Fleet Operational State Distribution")


    # Panel 4: Fleet State of Charge (SoC) Trajectory Band
    ax_soc = axs[1, 1]
    soc_stats = df.groupby('Time_Step')['SoC'].agg(['mean', 'min', 'max']).reset_index()
    ax_soc.plot(soc_stats['Time_Step'], soc_stats['mean'] * 100, color='purple', label='Mean Fleet SoC (%)', linewidth=1.5)
    ax_soc.fill_between(
        soc_stats['Time_Step'], soc_stats['min'] * 100, soc_stats['max'] * 100,
        color='purple', alpha=0.2, label='Min-Max SoC Band'
    )
    ax_soc.axhline(25, color='orange', linestyle='--', label='Min Dispatch SoC (25%)')
    ax_soc.set_title("Fleet Battery State of Charge (SoC) Trajectory")
    ax_soc.set_xlabel("Time Step (Hours)")
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_ylim(0, 105)
    ax_soc.legend(loc="lower right")
    ax_soc.grid(True, alpha=0.2)

    dashboard_png = f"{base_name}_dashboard.png"
    plt.tight_layout()
    plt.savefig(dashboard_png, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[PLOT SAVED] -> {dashboard_png}\n")

if __name__ == "__main__":
    csv_input = sys.argv[1] if len(sys.argv) > 1 else 'test_eval.csv'
    if os.path.exists(csv_input):
        analyze_and_plot(csv_input)
    else:
        print(f"Error: {csv_input} not found.")