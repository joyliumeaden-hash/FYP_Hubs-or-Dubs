import numpy as np
import pylab as pl
import os

from netCDF4 import Dataset

import xarray as xr

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

#Please change the directory 

directory_main = "C:/Users/joyli//OneDrive/Desktop/FYP/"
loc_data = pd.read_csv(directory_main + "/data/fyp_location.csv", encoding = "latin1", dtype={"Lat":str, "Lon":str})
location = loc_data["Project Name"] #.tolist()

numLocs = len(location)

base_years  = [2019, 2020, 2021, 2022, 2023, 2024]  # folders named (year)_Results
data_years  = [2019, 2020, 2021, 2022, 2023, 2024]  # columns inside each CSV
 
 
# ── What to extract ───────────────────────────────────────────────────────────
# For each location and each base year folder, extract:
# - total deficit MWh per data year
# - deficit hours per data year
# - total generation per data year
 
summary_rows = []
 
for base_yr in base_years:
    folder = directory_main + str(base_yr) + "_Results/"
 
    for loc in range(numLocs):
        loc_name = location[loc]
        # match your filename format: deficit_LocationName_.csv
        filename = folder + "deficit_" + loc_name + ".csv"
 
        if not os.path.exists(filename):
            print(f"Missing: {filename}")
            continue
 
        df = pd.read_csv(filename)
 
        for data_yr in data_years:
            deficit_col = f"{data_yr} deficit (MW)"
            gen_col     = f"{data_yr} generation (MW)"
            storage_col = f"{data_yr} storage (MW)"
 
            if deficit_col not in df.columns:
                continue
 
            deficit_series  = df[deficit_col].values
            gen_series      = df[gen_col].values if gen_col in df.columns else np.zeros(len(df))
            storage_series  = df[storage_col].values if storage_col in df.columns else np.zeros(len(df))
 
            total_deficit   = round(deficit_series.sum(), 2)
            deficit_hours   = int(np.sum(deficit_series > 0))
            total_gen       = round(gen_series.sum(), 2)
            avg_storage     = round(storage_series.mean(), 2)
 
            summary_rows.append({
                'base_year':      base_yr,
                'location':       loc_name,
                'data_year':      data_yr,
                'total_deficit_mwh': total_deficit,
                'deficit_hours':  deficit_hours,
                'total_gen_mwh':  total_gen,
                'avg_storage_mwh': avg_storage
            })
 
        print(f"  Extracted: base={base_yr} | {loc_name}")
 
# ── Save summary ──────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
output_path = directory_main + "Analysis/all_results_summary.csv"
os.makedirs(directory_main + "Analysis/", exist_ok=True)
summary_df.to_csv(output_path, index=False)
 
print(f"\nDone. {len(summary_rows)} rows saved to: {output_path}")
print(summary_df.head(10).to_string(index=False))
 
