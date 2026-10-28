from pathlib import Path
import pandas as pd

# Get the directory where this script is located, then go up one level to the project root
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / 'data' / 'nyc_demand.csv'

df = pd.read_csv(csv_path)

# 15 high-density Manhattan TLC Zones (The Microgrid)
cluster_zones = [161, 162, 163, 170, 236, 237, 141, 142, 143, 239, 140, 137, 107, 224, 229]

# Keep only rides that start AND end within our 15-node microgrid
df_filtered = df[df['pickup_location'].isin(cluster_zones) & df['dropoff_location'].isin(cluster_zones)]

output_path = script_dir.parent / 'data' / 'nyc_demand_15_nodes.csv'
df_filtered.to_csv(output_path, index=False)
print(f"Dataset shrunk from {len(df)} rows to {len(df_filtered)} rows.")