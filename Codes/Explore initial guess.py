import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# ── Directories ───────────────────────────────────────────────────────────────
directory_main = "C:/Users/joyli//OneDrive/Desktop/FYP/"

# ── Location to test (hardcoded index from fyp_location.csv) ─────────────────
LOC_INDEX = 4
BASE_YEAR  = "2024"

# ── Demand profile ────────────────────────────────────────────────────────────
requiredMWh = []
for day in range(365):
    for hour in range(24):
        if 8 <= hour < 10:
            requiredMWh.append(25)
        elif 10 <= hour < 18:
            requiredMWh.append(100)
        else:
            requiredMWh.append(0)
# data starts at 11 AM
requiredMWh = requiredMWh[11:] + requiredMWh[:11]
requiredMWh = np.array(requiredMWh)

# ── Load location metadata ────────────────────────────────────────────────────
loc_data  = pd.read_csv(directory_main + "/data/fyp_location.csv",
                        encoding="latin1", dtype={"Lat": str, "Lon": str})
lat_val   = loc_data["Lat"]
lon_val   = loc_data["Lon"]
text_val  = loc_data["Project Name"]
state_val = loc_data["State"]

# ── Load solar and wind CF data for the chosen location ───────────────────────
filename_solar = (directory_main + str(state_val[LOC_INDEX]) + " data/" +
                  BASE_YEAR + "_pv_"   + str(lat_val[LOC_INDEX]) + "_" + str(lon_val[LOC_INDEX]) + ".csv")
filename_wind  = (directory_main + str(state_val[LOC_INDEX]) + " data/" +
                  BASE_YEAR + "_wind_" + str(lat_val[LOC_INDEX]) + "_" + str(lon_val[LOC_INDEX]) + ".csv")

solarDataraw = pd.read_csv(filename_solar, skiprows=3)
windDataraw  = pd.read_csv(filename_wind,  skiprows=3)

solarCF = solarDataraw.to_xarray()["electricity"].values / 0.9
windCF  = windDataraw.to_xarray()["electricity"].values  / 0.9

solarCF = solarCF[:8760]
windCF  = windCF[:8760]

# reshape to (1, 8760) to match expected input shape
solarCFs_loop = solarCF[np.newaxis, :]
windCFs_loop  = windCF[np.newaxis, :]

# ── Cost functions ────────────────────────────────────────────────────────────
def calculate_annuity_factor(lifetime, discount_rate):
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime))

def PlantAnnualizedCostFunction(solarMW, windMW, batteryMW):
    capexCostPerMW_solar   = 1.414e6
    capexCostPerMW_wind    = 3.113e6
    capexCostPerMW_battery = 1.652e6

    capexCost_solar   = capexCostPerMW_solar   * np.sum(solarMW)
    capexCost_wind    = capexCostPerMW_wind    * np.sum(windMW)
    capexCost_battery = capexCostPerMW_battery * batteryMW

    discount_rate = 0.0599
    af = lambda life: calculate_annuity_factor(life, discount_rate)
    om = lambda capex: 0.015 * capex

    return (af(25) * capexCost_solar +
            af(25) * capexCost_wind  +
            af(10) * capexCost_battery +
            om(capexCost_solar) + om(capexCost_wind) + om(capexCost_battery))

def CalculatePowerDeficit(solarCFs, windCFs, requiredMWh, solarMW, windMW, batteryMW):
    hours      = len(requiredMWh)
    batteryMWh = 4 * batteryMW
    batteryEff = 0.8

    totalMWh = (np.sum(solarCFs * solarMW[:, np.newaxis], axis=0) +
                np.sum(windCFs  * windMW[:, np.newaxis],  axis=0))

    batteryStorage   = np.zeros(hours)
    batteryStorage[0] = batteryMWh / 2
    powerDeficitMWh  = np.zeros(hours)
    deficit_count    = 0

    for i in range(hours):
        prev_level = batteryStorage[i-1] if i > 0 else batteryStorage[0]

        if requiredMWh[i] == 0:
            excess = totalMWh[i] * batteryEff
            charge = min(excess, batteryMW)
            batteryStorage[i] = min(prev_level + charge, batteryMWh)
            powerDeficitMWh[i] = 0

        elif totalMWh[i] > requiredMWh[i]:
            powerDeficitMWh[i] = 0
            excess = (totalMWh[i] - requiredMWh[i]) * batteryEff
            charge = min(excess, batteryMW)
            batteryStorage[i] = min(prev_level + charge, batteryMWh)

        else:
            discharge = min(prev_level, batteryMW)
            available = totalMWh[i] + discharge
            if available > requiredMWh[i]:
                discharge = requiredMWh[i] - totalMWh[i]
            powerDeficitMWh[i] = np.maximum(requiredMWh[i] - totalMWh[i] - discharge, 0.0)
            batteryStorage[i]  = prev_level - discharge
            if powerDeficitMWh[i] > 0:
                deficit_count += 1

    return powerDeficitMWh, deficit_count, batteryStorage, totalMWh

# ── Optimization function ─────────────────────────────────────────────────────
def run_optimization(solarCFs, windCFs, requiredMWh, initial_guess_wind, initial_guess_solar, initial_guess_battery):
    numSolar    = solarCFs.shape[0]
    numWind     = windCFs.shape[0]
    solarOffset = 0
    windOffset  = solarOffset + numSolar
    batteryIndx = windOffset  + numWind
    deficitPenalty = 20000

    def calculateAnnualizedCost(params):
        solarMWs  = params[solarOffset:solarOffset + numSolar]
        windMWs   = params[windOffset:windOffset   + numWind]
        batteryMW = params[batteryIndx]
        deficit   = CalculatePowerDeficit(solarCFs, windCFs, requiredMWh,
                                          solarMWs, windMWs, batteryMW)[0]
        return PlantAnnualizedCostFunction(solarMWs, windMWs, batteryMW) + np.sum(deficit) * deficitPenalty

    xinit = np.zeros(numSolar + numWind + 1)
    xinit[:numSolar]              = initial_guess_solar
    xinit[numSolar:numSolar+numWind] = initial_guess_wind
    xinit[batteryIndx]            = initial_guess_battery

    #bounds = [[0, None]] * (numSolar + numWind + 1)
    #res    = minimize(calculateAnnualizedCost, xinit, bounds=bounds)#, method='Nelder-Mead')

    bounds = [(0, 1000)] * (numSolar + numWind + 1)
    res = differential_evolution(calculateAnnualizedCost, bounds, 
                              seed=42, maxiter=1000, tol=1e-6,
                              workers=1, polish=True)

    solarMWs  = res.x[solarOffset:solarOffset + numSolar]
    windMWs   = res.x[windOffset:windOffset   + numWind]
    batteryMW = res.x[batteryIndx]

    _, d_count, _, _ = CalculatePowerDeficit(solarCFs, windCFs, requiredMWh,
                                              solarMWs, windMWs, batteryMW)
    plant_cost   = PlantAnnualizedCostFunction(solarMWs, windMWs, batteryMW)
    penalty_cost = res.fun - plant_cost
    lcoe         = res.fun / np.sum(requiredMWh)

    return {
        'ini_solar': initial_guess_solar,
        'ini_wind': initial_guess_wind,
        'ini_battery': initial_guess_battery,
        'solar_mw':      round(solarMWs[0],  2),
        'wind_mw':       round(windMWs[0],   2),
        'battery_mw':    round(batteryMW,    2),
        'annualized_cost': round(res.fun,    2),
        'plant_cost':    round(plant_cost,   2),
        'penalty_cost':  round(penalty_cost, 2),
        'lcoe':          round(lcoe,         2),
        'deficit_hours': d_count,
        'converged':     res.success,
        'message':       res.message
    }


# ── Multi-start exploration ───────────────────────────────────────────────────
initial_guesses_wind = [200]
initial_guesses_solar = [200]
initial_guesses_battery = [200]

print(f"Running multi-start exploration for: {text_val[LOC_INDEX]}")
print(f"Base year: {BASE_YEAR}\n")

multistart_results = []

for guess_wind in initial_guesses_wind:
        for guess_solar in initial_guesses_solar:
            for guess_battery in initial_guesses_battery:
                #print(f"  Running with initial guess: {guess} MW ...", flush=True)
                result = run_optimization(solarCFs_loop, windCFs_loop, requiredMWh, guess_wind,guess_solar,guess_battery)
                multistart_results.append(result)
                print(f"    Solar: {result['solar_mw']} MW | Wind: {result['wind_mw']} MW | "
                    f"Battery: {result['battery_mw']} MW | LCOE: ${result['lcoe']:.2f} | "
                    f"Deficit hours: {result['deficit_hours']}")

# ── Save results ──────────────────────────────────────────────────────────────
df = pd.DataFrame(multistart_results)
print("\n── Summary ──────────────────────────────────────────────────────────")
print(df.to_string(index=False))

print(f"\nCost range:   ${df['annualized_cost'].min():,.0f}  to  ${df['annualized_cost'].max():,.0f}")
#print(f"Solar range:  {df['solar_mw'].min()} MW  to  {df['solar_mw'].max()} MW")
#print(f"Wind range:   {df['wind_mw'].min()} MW  to  {df['wind_mw'].max()} MW")
#print(f"Battery range:{df['battery_mw'].min()} MW  to  {df['battery_mw'].max()} MW")

#output_path = directory_main + "single_test/multistart_results_ini" + BASE_YEAR + ".csv"
#df.to_csv(output_path, index=False)
#print(f"\nResults saved to: {output_path}")

"""







# ── Multi-year robustness section ─────────────────────────────────────────────
# Hardcode a plant arrangement to test across all years
# Change these values to whatever arrangement you want to explore
HARDCODED_SOLAR_MW   = 320.73  # MW
HARDCODED_WIND_MW    = 53.73   # MW
HARDCODED_BATTERY_MW = 90.56   # MW

# Years to test
data_years = [2019, 2020, 2021, 2022, 2023, 2024]
 
print(f"\n\n── Multi-year robustness for hardcoded arrangement ──────────────────")
print(f"Solar: {HARDCODED_SOLAR_MW} MW | Wind: {HARDCODED_WIND_MW} MW | Battery: {HARDCODED_BATTERY_MW} MW")
print(f"Location: {text_val[LOC_INDEX]}\n")
 
# Load all years of CF data for this location
solarCF_future = []
windCF_future  = []
 
for yr in data_years:
    fname_solar = (directory_main + str(state_val[LOC_INDEX]) + " data/" +
                   str(yr) + "_pv_"   + str(lat_val[LOC_INDEX]) + "_" + str(lon_val[LOC_INDEX]) + ".csv")
    fname_wind  = (directory_main + str(state_val[LOC_INDEX]) + " data/" +
                   str(yr) + "_wind_" + str(lat_val[LOC_INDEX]) + "_" + str(lon_val[LOC_INDEX]) + ".csv")
 
    sD = pd.read_csv(fname_solar, skiprows=3).to_xarray()["electricity"].values / 0.9
    wD = pd.read_csv(fname_wind,  skiprows=3).to_xarray()["electricity"].values / 0.9
 
    solarCF_future.append(sD[:8760])
    windCF_future.append(wD[:8760])
 
solarCF_future = np.array(solarCF_future)
windCF_future  = np.array(windCF_future)
 
# Run through each year with the hardcoded arrangement
solarMWs_hc  = np.array([HARDCODED_SOLAR_MW])
windMWs_hc   = np.array([HARDCODED_WIND_MW])
batteryMW_hc = HARDCODED_BATTERY_MW
 
plant_cost_hc = PlantAnnualizedCostFunction(solarMWs_hc, windMWs_hc, batteryMW_hc)
 
all_rows   = []
row_labels = []
year_summary = []
 
for j, yr in enumerate(data_years):
    solarCF_floop = solarCF_future[[j], :]
    windCF_floop  = windCF_future[[j],  :]
 
    deficit, d_count, storage, totgen = CalculatePowerDeficit(
        solarCF_floop, windCF_floop, requiredMWh,
        solarMWs_hc, windMWs_hc, batteryMW_hc)
 
    penalty_cost  = np.sum(deficit) * 20000
    total_cost    = plant_cost_hc + penalty_cost
    lcoe          = total_cost / np.sum(requiredMWh)
 
    all_rows.append(deficit)
    all_rows.append(totgen)
    all_rows.append(storage)
 
    row_labels.append(f"{yr} deficit (MWh)")
    row_labels.append(f"{yr} generation (MWh)")
    row_labels.append(f"{yr} storage (MWh)")
 
    year_summary.append({
        'year':          yr,
        'deficit_hours': d_count,
        'total_deficit_mwh': round(np.sum(deficit), 2),
        'plant_cost':    round(plant_cost_hc, 2),
        'penalty_cost':  round(penalty_cost,  2),
        'total_cost':    round(total_cost,    2),
        'lcoe':          round(lcoe,          2)
    })
 
    print(f"  {yr} | Deficit hours: {d_count:3d} | Total deficit: {np.sum(deficit):8.1f} MWh | LCOE: ${lcoe:.2f}")
 
# Save hourly data (deficit, generation, storage) per year
#hourly_df = pd.DataFrame(all_rows, index=row_labels).T
#hourly_df.to_csv(directory_main + "single_test/multiyear_hourly.csv", index=False)
 
# Save year summary
summary_df = pd.DataFrame(year_summary)
summary_df.to_csv(directory_main + "single_test/multiyear_summary.csv", index=False)
 
print(f"\nHourly data saved to:  single_test/multiyear_hourly.csv")
print(f"Year summary saved to: single_test/multiyear_summary.csv")
"""