# import enum
# import os
# import pandas as pd
# import numpy as np

# # Monkey-patch StrEnum for Python 3.10 compatibility before gridstatus loads
# if not hasattr(enum, 'StrEnum'):
#     class StrEnum(str, enum.Enum):
#         pass
#     enum.StrEnum = StrEnum

# import gridstatus

# class PricingUtility:
#     """Fetches or mocks Locational Marginal Pricing (LMP) for simulated cities."""
    
#     def __init__(self, city, start_date, end_date):
#         self.city = city
#         self.start_date = start_date
#         self.end_date = end_date
        
#         # New split logic: Keep mode as a string indicator, and store dataframe separately
#         self.pricing_mode = None
#         self.df_data = None
        
#         self._load_or_fetch_data()



#     # def _load_or_fetch_data(self):
#     #     """Attempts to load grid data from APIs; falls back to synthetic curves on failure."""
#     #     if self.city.lower() == 'chicago':
#     #         if 'PJM_API_KEY' not in os.environ:
#     #             print("WARNING: PJM_API_KEY missing. Using dynamic synthetic ComEd LMP.")
#     #             self.pricing_mode = "SYNTHETIC_COMED_CURVE"
#     #             return
#     #         try:
#     #             iso = gridstatus.PJM()
#     #             self.df_data = iso.get_lmp(
#     #                 date=self.start_date,
#     #                 end=self.end_date,
#     #                 market="REAL_TIME_5_MIN",
#     #                 locations=["COMED"]
#     #             )
#     #             self.pricing_mode = "LIVE_DATA"
#     #         except Exception as e:
#     #             print(f"PJM API Error: {e}. Falling back to synthetic curve.")
#     #             self.pricing_mode = "SYNTHETIC_COMED_CURVE"
                
#     #     elif self.city.lower() == 'nyc':
#     #         try:
#     #             iso = gridstatus.NYISO()
#     #             # Fetches 5-minute real-time LMP for the N.Y.C. load zone
#     #             self.df_data = iso.get_lmp(
#     #                 date=self.start_date,
#     #                 end=self.end_date,
#     #                 market="REAL_TIME_5_MIN",
#     #                 locations=["N.Y.C."],
#     #                 location_type="zone"
#     #             )
#     #             self.pricing_mode = "LIVE_DATA"
#     #         except Exception as e:
#     #             print(f"NYISO API Error: {e}. Falling back to synthetic NYC curve.")
#     #             self.pricing_mode = "SYNTHETIC_NYC_CURVE"
        
#     #     else:
#     #         raise ValueError(f"Unsupported city configuration: {self.city}")

#     # def get_price(self, timestamp):
#     #     """Returns the grid price in $/kWh for a given timestamp."""
#     #     t_hour = timestamp.hour + (timestamp.minute / 60.0)
        
#     #     if self.pricing_mode == "SYNTHETIC_COMED_CURVE":
#     #         return 0.03 + 0.15 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))
            
#     #     elif self.pricing_mode == "SYNTHETIC_NYC_CURVE":
#     #         return 0.05 + 0.25 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))

#     #     else:
#     #         # Parses the live pandas DataFrame from gridstatus
#     #         try:
#     #             df_tz = self.df_data['Interval Start'].dt.tz
#     #             ts = pd.to_datetime(timestamp).tz_localize(df_tz)
                
#     #             row = self.df_data[
#     #                 (self.df_data['Interval Start'] <= ts) & 
#     #                 (self.df_data['Interval End'] > ts)
#     #             ]
                
#     #             if not row.empty:
#     #                 # 🟢 VERIFICATION 1: Print the live price to confirm it's working
#     #                 price = float(row.iloc[0]['LMP']) / 1000.0
#     #                 print(f"[LIVE ENERGY PRICE] Timestamp: {timestamp} -> Price: ${price:.4f}/kWh")
#     #                 return price
#     #             else:
#     #                 # 🔴 VERIFICATION 2: Caught empty data windows
#     #                 fallback = 0.05 if self.city.lower() == 'nyc' else 0.03
#     #                 print(f"[FALLBACK WARNING] No data row found for {timestamp}. Using baseline: ${fallback}/kWh")
#     #                 return fallback
#     #         except Exception as e:
#     #             # 🔴 VERIFICATION 3: Caught parsing/timezone errors
#     #             fallback = 0.05 if self.city.lower() == 'nyc' else 0.03
#     #             print(f"[CRITICAL FALLBACK] Error parsing live DataFrame: {e}. Using baseline: ${fallback}/kWh")
#     #             return fallback

import enum
import os
import pandas as pd
import numpy as np

# Monkey-patch StrEnum for Python 3.10 compatibility before gridstatus loads
if not hasattr(enum, 'StrEnum'):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

class PricingUtility:
    def __init__(self, city, start_date, end_date):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        # CRITICAL FIX: Explicitly initialize pricing_data so get_price() can access it
        self.pricing_data = self._load_or_fetch_data()

    def _load_or_fetch_data(self):
        import gridstatus
        
        # 1. Chicago Logic
        if self.city.lower() == 'chicago':
            if 'PJM_API_KEY' not in os.environ:
                print("WARNING: PJM Data Miner unreachable. Using synthetic ComEd LMP.")
                return "SYNTHETIC_COMED_CURVE"
            try:
                iso = gridstatus.PJM()
                return iso.get_lmp(date=self.start_date, end=self.end_date, market="REAL_TIME_5_MIN", locations=["COMED"])
            except Exception as e:
                print(f"PJM API Error: {e}. Falling back to synthetic curve.")
                return "SYNTHETIC_COMED_CURVE"
                
        # 2. NYC Logic (Cache-First Implementation)
        elif self.city.lower() == 'nyc':
            fallback_dir = "fallback_nyc"
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_file = os.path.join(fallback_dir, f"nyc_pricing_{self.start_date}_{self.end_date}.csv")
            
            # OPTIMIZATION: Check for the local file FIRST before ever touching the API
            if os.path.exists(fallback_file):
                print(f"Local cache found! Loading real NYC grid data from {fallback_file}...")
                return pd.read_csv(fallback_file)
            
            # If the file doesn't exist, fetch it once and save it forever
            try:
                print(f"No local cache found. Fetching live NYISO data from {self.start_date} to {self.end_date}...")
                iso = gridstatus.NYISO()
                df = iso.get_lmp(
                    date=self.start_date,
                    end=self.end_date,
                    market="REAL_TIME_5_MIN",
                    locations=["N.Y.C."]
                )
                
                df.to_csv(fallback_file, index=False)
                print(f"Successfully fetched and cached real NYC grid data to: {fallback_file}")
                return df
                
            except Exception as e:
                print(f"NYISO API Connection Error: {e}")
                print("No local backup exists yet. Falling back to synthetic NYC curve.")
                return "SYNTHETIC_NYC_CURVE"

    def get_price(self, timestamp):
        # Fallback 1: Chicago Synthetic
        if isinstance(self.pricing_data, str) and self.pricing_data == "SYNTHETIC_COMED_CURVE":
            t_hour = timestamp.hour + (timestamp.minute / 60.0)
            return 0.03 + 0.15 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))
            
        # Fallback 2: NYC Synthetic 
        if isinstance(self.pricing_data, str) and self.pricing_data == "SYNTHETIC_NYC_CURVE":
            t_hour = timestamp.hour + (timestamp.minute / 60.0)
            return 0.05 + 0.20 * np.exp(-0.5 * ((t_hour - 18.0) / 2.0) ** 2)
            
        # Primary Logic: Real Grid Data (From API or Local Backup)
        try:
            df = self.pricing_data
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Time' in df.columns:
                    df['Time'] = pd.to_datetime(df['Time'])
                    df.set_index('Time', inplace=True)
            
            query_ts = pd.Timestamp(timestamp)
            if df.index.tz is not None and query_ts.tz is None:
                query_ts = query_ts.tz_localize('US/Eastern')
                
            nearest_idx = df.index.get_indexer([query_ts], method='nearest')[0]
            return float(df.iloc[nearest_idx]['LMP'])
        except Exception:
            return 0.05