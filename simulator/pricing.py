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
    """Fetches or mocks Locational Marginal Pricing (LMP) for simulated cities."""

    def __init__(self, city, start_date, end_date):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.pricing_data = self._load_or_fetch_data()

    def _prepare_dataframe(self, df):
        """Standardizes timestamps, timezones, and sorts index ONCE for O(log N) lookup."""
        if not isinstance(df, pd.DataFrame):
            return df

        # Handle 'Time' or 'Interval Start' column from gridstatus
        time_col = None
        for col in ['Time', 'Interval Start', 'interval_start', 'timestamp']:
            if col in df.columns:
                time_col = col
                break

        if time_col is not None:
            # Parse with utc=True to avoid mixed timezone bugs, then convert to US/Eastern
            df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert('US/Eastern')
            df.set_index(time_col, inplace=True)

        # Sort the DatetimeIndex so get_indexer(method='nearest') runs in microseconds
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize('US/Eastern')
            df.sort_index(inplace=True)

        return df

    def _load_or_fetch_data(self):
        import gridstatus

        # 1. Chicago Logic
        if self.city.lower() == 'chicago':
            if 'PJM_API_KEY' not in os.environ:
                print("WARNING: PJM Data Miner unreachable. Using synthetic ComEd LMP.")
                return "SYNTHETIC_COMED_CURVE"
            try:
                iso = gridstatus.PJM()
                df = iso.get_lmp(
                    date=self.start_date,
                    end=self.end_date,
                    market="REAL_TIME_5_MIN",
                    locations=["COMED"]
                )
                return self._prepare_dataframe(df)
            except Exception as e:
                print(f"PJM API Error: {e}. Falling back to synthetic curve.")
                return "SYNTHETIC_COMED_CURVE"

        # 2. NYC Logic (Cache-First Implementation)
        elif self.city.lower() == 'nyc':
            fallback_dir = "fallback_nyc"
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_file = os.path.join(fallback_dir, f"nyc_pricing_{self.start_date}_{self.end_date}.csv")

            # Check for the local cached file FIRST
            if os.path.exists(fallback_file):
                print(f"Local cache found! Loading real NYC grid data from {fallback_file}...")
                df = pd.read_csv(fallback_file)
                return self._prepare_dataframe(df)

            # If the file doesn't exist, fetch it once and save it
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
                return self._prepare_dataframe(df)

            except Exception as e:
                print(f"NYISO API Connection Error: {e}")
                print("No local backup exists yet. Falling back to synthetic NYC curve.")
                return "SYNTHETIC_NYC_CURVE"

        else:
            raise ValueError(f"Unsupported city configuration: {self.city}")

    def get_price(self, timestamp):
        """Returns the grid price in $/kWh (or $/MWh) for a given timestamp."""
        # Fallback 1: Chicago Synthetic
        if isinstance(self.pricing_data, str) and self.pricing_data == "SYNTHETIC_COMED_CURVE":
            t_hour = timestamp.hour + (timestamp.minute / 60.0)
            return 0.03 + 0.15 * max(0, np.sin((np.pi / 12.0) * (t_hour - 6.0)))

        # Fallback 2: NYC Synthetic 
        if isinstance(self.pricing_data, str) and self.pricing_data == "SYNTHETIC_NYC_CURVE":
            t_hour = timestamp.hour + (timestamp.minute / 60.0)
            return 0.05 + 0.20 * np.exp(-0.5 * ((t_hour - 18.0) / 2.0) ** 2)

        # Primary Logic: Fast Binary Search on Prepared DataFrame
        try:
            query_ts = pd.Timestamp(timestamp)
            if query_ts.tz is None:
                query_ts = query_ts.tz_localize('US/Eastern')
            else:
                query_ts = query_ts.tz_convert('US/Eastern')

            # Fast binary search nearest lookup
            nearest_idx = self.pricing_data.index.get_indexer([query_ts], method='nearest')[0]
            price = float(self.pricing_data.iloc[nearest_idx]['LMP'])
            
            return price
        except Exception:
            return 0.05
