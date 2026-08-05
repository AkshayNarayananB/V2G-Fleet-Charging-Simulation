"""Pricing utility to manage real-time electricity rates for V2B arbitrage."""

import datetime
import os
import pandas as pd
import gridstatus

class PricingUtility:
    def __init__(self, city: str, start_date: str, end_date: str):
        """
        Args:
            city: 'chicago' or 'nyc'
            start_date: YYYY/MM/DD format
            end_date: YYYY/MM/DD format
        """
        self.city = city.lower()
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        self.pricing_data = self._load_or_fetch_data()

    def _load_or_fetch_data(self) -> pd.DataFrame:
        # cache_file = f"../data/{self.city}_pricing_{self.start.year}.csv"
        cache_file = f"data/{self.city}_pricing_{self.start.year}.csv"
        
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['interval_start'])
            df.set_index('interval_start', inplace=True)
            return df

        print(f"Fetching historical grid data for {self.city}...")
        
        # if self.city == 'nyc':
        #     iso = gridstatus.NYISO()
        #     # NYISO Zone J covers New York City
        #     df = iso.get_historical_lmp("DAY_AHEAD_HR", start=self.start, end=self.end)
        #     df = df[df['Location'] == 'N.Y.C.']
            
        # elif self.city == 'chicago':
        #     iso = gridstatus.PJM()
        #     # PJM COMED zone covers Chicago
        #     df = iso.get_historical_lmp("DAY_AHEAD_HR", start=self.start, end=self.end)
        #     df = df[df['Location'] == 'COMED']

        if self.city == 'nyc':
            iso = gridstatus.NYISO()
            # NYISO Zone J covers New York City
            df = iso.get_lmp(
                market="DAY_AHEAD_HOURLY", 
                start=self.start, 
                end=self.end, 
                locations=["N.Y.C."]
            )
            
        elif self.city == 'chicago':
            iso = gridstatus.PJM()
            # PJM COMED zone covers Chicago
            df = iso.get_lmp(
                market="DAY_AHEAD_HOURLY", 
                start=self.start, 
                end=self.end, 
                locations=["COMED"]
            )
        else:
            raise ValueError("City must be 'chicago' or 'nyc'")

        # Standardize timezone to UTC to match taxi data, convert $/MWh to $/kWh
        df['interval_start'] = pd.to_datetime(df['Interval Start']).dt.tz_convert('UTC').dt.tz_localize(None)
        df['price_per_kwh'] = df['LMP'] / 1000.0 
        
        # Select required columns and cache
        clean_df = df[['interval_start', 'price_per_kwh']].set_index('interval_start')
        clean_df.to_csv(cache_file)
        return clean_df

    def get_price(self, current_time: datetime.datetime) -> float:
        """Return the electricity price ($/kWh) for the current simulation hour."""
        # Floor the timestamp to the nearest hour to match day-ahead market intervals
        hour_start = current_time.replace(minute=0, second=0, microsecond=0)
        try:
            return self.pricing_data.loc[hour_start]['price_per_kwh']
        except KeyError:
            # Fallback average price if specific hour is missing from grid API
            return 0.15