import pandas as pd
import requests
import time

def download_taxi_data(endpoint, start_date, end_date, output_file):
    print(f"Starting download for {start_date} to {end_date}...")
    base_url = f"https://data.cityofchicago.org/resource/{endpoint}.csv"
    
    limit = 50000
    offset = 0
    all_chunks = []
    
    while True:
        # Format the SoQL query
        query = (
            f"?$where=trip_start_timestamp>='{start_date}T00:00:00' "
            f"AND trip_start_timestamp<'{end_date}T00:00:00'"
            f"&$limit={limit}&$offset={offset}&$order=trip_start_timestamp"
        )
        url = base_url + query
        
        print(f"Fetching rows {offset} to {offset + limit}...")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
            
        # Read the CSV chunk from the response
        from io import StringIO
        chunk = pd.read_csv(StringIO(response.text))
        
        if chunk.empty:
            print("Finished downloading all records.")
            break
            
        all_chunks.append(chunk)
        offset += limit
        time.sleep(1) # Be nice to the API

    if all_chunks:
        final_df = pd.concat(all_chunks, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} rows to {output_file}")

# Download 2020-2023 (Legacy Endpoint)
download_taxi_data("wrvz-psew", "2020-01-01", "2024-01-01", "chicago_2020_2023.csv")

# Download 2024-Present (New Endpoint)
download_taxi_data("ajtu-isnz", "2024-01-01", "2024-12-31", "chicago_2024_present.csv")
