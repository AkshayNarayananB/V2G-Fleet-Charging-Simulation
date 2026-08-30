import os
import requests

def download_nyc_data(start_year, end_year, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_str = f"{month:02d}"
            filename = f"yellow_tripdata_{year}-{month_str}.parquet"
            # TLC CloudFront URL pattern
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
            output_path = os.path.join(output_dir, filename)
            
            print(f"Downloading NYC {year}-{month_str}...")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f" -> Saved {filename}")
            else:
                print(f" -> File not found or unavailable: {response.status_code}")

download_nyc_data(2024, 2026)
