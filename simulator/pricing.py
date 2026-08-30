import enum
import os
import pandas as pd
import numpy as np

# Monkey-patch StrEnum for Python 3.10 compatibility before gridstatus loads
if not hasattr(enum, 'StrEnum'):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

import gridstatus

class PricingUtility:
    """Fetches or mocks Locational Marginal Pricing (LMP) for simulated cities."""
    
    def __init__(self, city, start_date, end_date):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.pricing_data = self._load_or_fetch_data()

    def _load_or_fetch_data(self):
        """Attempts to load grid data from APIs; falls back to synthetic curves on failure."""
        if self.city.lower() == 'chicago':
            if 'PJM_API_KEY' not in os.environ:
                print("WARNING: PJM_API_KEY missing. Using dynamic synthetic ComEd LMP.")
                return "SYNTHETIC_COMED_CURVE"
            try:
                iso = gridstatus.PJM()
                return iso.get_lmp(
                    date=self.start_date,
                    end=self.end_date,
                    market="REAL_TIME_5_MIN",
                    locations=["COMED"]
                )
            except Exception as e:
                print(f"PJM API Error: {e}. Falling back to synthetic curve.")
                return "SYNTHETIC_COMED_CURVE"
                
        elif self.city.lower() == 'nyc':
            try:
                iso = gridstatus.NYISO()
                # Fetches 5-minute real-time LMP for the N.Y.C. load zone
                return iso.get_lmp(
                    date=self.start_date,
                    end=self.end_date,
                    market="REAL_TIME_5_MIN",
                    locations=["N.Y.C."],
                    location_type="zone"
                )
            except Exception as e:
                print(f"NYISO API Error: {e}. Falling back to synthetic NYC curve.")
                return "SYNTHETIC_NYC_CURVE"
        
        else:
            raise ValueError(f"Unsupported city configuration: {self.city}")

    def get_price(self, timestamp):
        """Returns the grid price in $/kWh for a given timestamp."""
        t_hour = timestamp.hour + (timestamp.minute / 60.0)
        
        if self.pricing_data == "SYNTHETIC_COMED_CURVE":
            # Mimics Chicago: $0.03 base, spiking to $0.18 at 6:00 PM
            base_price = 0.03 
            peak_spike = 0.15 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))
            return base_price + peak_spike
            
        elif self.pricing_data == "SYNTHETIC_NYC_CURVE":
            # Mimics NYC: $0.05 base, spiking to $0.30 at 6:00 PM
            base_price = 0.05 
            peak_spike = 0.25 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))
            return base_price + peak_spike

        else:
            # Parses the live pandas DataFrame from gridstatus
            try:
                # Localize simulator timestamp to match the API DataFrame timezone
                df_tz = self.pricing_data['Interval Start'].dt.tz
                ts = pd.to_datetime(timestamp).tz_localize(df_tz)
                
                row = self.pricing_data[
                    (self.pricing_data['Interval Start'] <= ts) & 
                    (self.pricing_data['Interval End'] > ts)
                ]
                
                if not row.empty:
                    # gridstatus returns $/MWh; divide by 1000 for $/kWh
                    return float(row.iloc[0]['LMP']) / 1000.0
                else:
                    return 0.03
            except Exception:
                return 0.03
