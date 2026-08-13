import numpy as np
import pylab as pl

from scipy.optimize import minimize

#Please change the directory 

directory = "C:/Users/joyli//OneDrive/Desktop/FYP/Coding and Data/"
directory_main = "C:/Users/joyli//OneDrive/Desktop/FYP/"


def calculate_annuity_factor(lifetime, discount_rate):
    #converts up front cost into yearly cost, used in below functions
    #lifetime = site lifetime in years? 
    #Discound rate = 
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime))             

#calculates the annual plant cost, considering CAPEX and Maintenance
def PlantAnnualizedCostFunction(solarMW, windMW, batteryMW):
    
    #CAPEX of each renewable energy component per MW
    capexCostPerMW_solar = 1.414e6
    capexCostPerMW_wind = 3.113e6
    capexCostPerMW_battery = 1.652e6
    
    #Total cost depending on the size of production
    capexCost_solar = capexCostPerMW_solar * np.sum(solarMW)
    capexCost_wind = capexCostPerMW_wind * np.sum(windMW)
    capexCost_battery = capexCostPerMW_battery * batteryMW
            
    discount_rate = 0.0599
    #lambda makes it that only the lifetime is neded for input
    af = lambda life: calculate_annuity_factor(life, discount_rate)     #annualised cost
    om = lambda capex: 0.015 * capex           #operation and maintenance cost
    
    #Calculates the total annual cost for all energy production method (CAPEX and Maintenance)
    totalAnnualizedCost = af(25) * capexCost_solar + \
        af(25) * capexCost_wind + \
        af(10) * capexCost_battery + \
        om(capexCost_solar) + om(capexCost_wind) + om(capexCost_battery) 
    return totalAnnualizedCost       #plus all together and gives out value
    
#Gives deficit of each hour if there is one
#SolarCFs, WindCFs = Capacity factors
#SolarMW, WindMW = size of solar/wind farm
#Battery MW = discharge rate of batteries
#RequiredMWh = demand
def CalculatePowerDeficit(solarCFs, windCFs, requiredMWh, solarMW, windMW, batteryMW):
    #hours = 8760     #number of hours in a year 
    hours = len(requiredMWh)
    batteryHours = 4     #battery duration how long they can be dischagred for
    batteryMWh = batteryHours * batteryMW   #how much energy they can provide each discharge
    batteryEff = 0.8    #batery efficiency
    
    #gives total amount of energy from wind and solar
    #axis=0 means calculating the column-wise sum
    #solarMW[:,np.newaxis] converts the MW data array to column form
    totalMWh = np.sum(solarCFs * solarMW[:,np.newaxis], axis = 0) + np.sum(windCFs * windMW[:,np.newaxis], axis = 0) 
    
    batteryStorage = np.zeros(hours)   #Create array filled with 0 the same length as hours in a year
    batteryStorage[0] = batteryMWh / 2    #This is assuming starting the battery at half full
    
    powerDeficitMWh = np.zeros(hours)   #stores the demand value as an array
    deficit_count = 0
    
    for i in range(hours):
        prev_level = batteryStorage[i-1] if i>0 else batteryStorage[0]

        if requiredMWh[i] == 0:
            # no demand
            excess = totalMWh[i] * batteryEff
            charge = min(excess, batteryMW)
            batteryStorage[i] = min(prev_level + charge, batteryMWh)
            powerDeficitMWh[i]=0
        #if the produced energy is higher than demand
        elif totalMWh[i] > requiredMWh[i]:       #goes through each hour
            powerDeficitMWh[i] = 0     #deficit would be 0 if there is excess energy
            excess = (totalMWh[i] - requiredMWh[i]) * batteryEff      #calculate the excess energy that will go into battery from each hour
            charge = min(excess, batteryMW)    #the charge will either be at th excess value or be at battery capacity
            #level = batteryStorage[i-1] + charge if i > 0 else batteryStorage[0] + charge #storage of last hour plus the excess? 
            batteryStorage[i] = min(prev_level + charge, batteryMWh)
        
        #if there is not enough energy to meet demand from wind and solar
        else:
            #available = totalMWh[i]  #get how much energy is avaliable from wind and solar from that hour
            #level = batteryStorage[i-1] if i > 0 else batteryStorage[0]        #battery storage from the hour before 
            discharge = min(prev_level, batteryMW)      #assuming discharging all avaliable bettery power
            available = totalMWh[i] +discharge
            if available > requiredMWh[i]:        #if more than required in that hour
                discharge = requiredMWh[i] - totalMWh[i]        #amount to discharge = demand - solar/wind energy from that hour
            #available += discharge        #total amount avaliable after calculation now also includes the dischargeable amount
            
            powerDeficitMWh[i] = np.maximum(requiredMWh[i] - totalMWh[i] - discharge,0.0)  #to calculate deficit if there is any (0 or deficit)
            batteryStorage[i] = prev_level - discharge      #avaliable energy level in battery after discharging it
            if powerDeficitMWh[i] > 0: 
                deficit_count=deficit_count+1
    
    return powerDeficitMWh, deficit_count, batteryStorage, totalMWh   #this gives a final deficit of each hour in an array


def run_single_optimization(solarCFs, windCFs, requiredMWh):
    """Run a single optimization and return results"""
    
    numSolar = solarCFs.shape[0]   #gets the number of rows/layer of data
    numWind = windCFs.shape[0]
    
    #150 is just a value 
    solarMWinit = 150*np.ones(numSolar)   #initial solar generation guess of 150MW
    windMWinit = 150*np.ones(numWind)   #initial wind generation
    batteryMWinit = 500                  #battery discharge rate? 
    
    solarOffset = 0
    windOffset = solarOffset + numSolar
    batteryIndx = windOffset + numWind
    
    deficitPenalty =  20000  # penalty per MWh missed. 
    
    #recalculate annualized cost
    #params = 
    def calculateAnnualizedCost(params):
        solarMWs = params[solarOffset:solarOffset+numSolar]  #gets the solar output from start to numSolar??
        
        windMWs = params[windOffset:windOffset+numWind]   #gets the wind output from when solarMWs stops plus numWind??
        
        batteryMW = params[batteryIndx]   #Just 1 value of the battery energy
        
        deficit = CalculatePowerDeficit(solarCFs, windCFs, requiredMWh, solarMWs, windMWs, batteryMW)[0]  #use defined function to calculate the deficit array

        cost = PlantAnnualizedCostFunction(solarMWs, windMWs, batteryMW) + np.sum(deficit) * deficitPenalty #the annualized cost + penalty
        return cost
    
    #res = minimize(calculateAnnualizedCost, [solarMWinit, windMWinit, batteryMWinit], method='Nelder-Mead')
    #final_solar, final_wind, final_batt = np.maximum(res.x, 0)
    
    xinit = np.zeros(numSolar+numWind+1)    #array with number of solar and number of wind data? 
    xinit[:numSolar] = solarMWinit          #fill initioal x value 0 -> solar data (in rows) with solar initial generation 
    xinit[numSolar:numSolar+numWind] = windMWinit      #Put in the initial wind guesses
    xinit[batteryIndx]  = batteryMWinit     #fill initial x value at battery index with initial battery discharge rate? 
    
    theBounds = []     #creating empty array for "theBounds"
    for i in range(numSolar+numWind+1):    #for the range in xinit    TO IMPROVE
      theBounds.append([0,None])     #This means theBounds right now = [0, None]
    
    #below uses scipy.optimize minimize function to find the minimal 
    res = minimize(calculateAnnualizedCost,xinit,bounds=theBounds, method='Nelder-Mead')#method ="COBYLA") # , method='Nelder-Mead')
    
    
    final_solar = res.x[solarOffset:solarOffset + numSolar]    #gets the x value out that represents final solar 
    final_wind = res.x[windOffset:windOffset + numWind]     #get the x values out that represents final wind 
    # fixme need wind heere
    final_batt = res.x[batteryIndx]        #get the value of battery index from result
    
    return {
        'solar_mw': final_solar,
        'wind_mw': final_wind,
        'battery_mw': final_batt,
        'anualizedCost': res.fun,
        'success': res.success,
        'message': res.message,
        'iterations': res.nit if hasattr(res, 'nit') else None
    }


################

# collect cf data at locations

#below are python libraries

from netCDF4 import Dataset

import xarray as xr

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

loc_data = pd.read_csv(directory_main + "/data/fyp_location.csv", encoding = "latin1", dtype={"Lat":str, "Lon":str}) #all location name, lat and lon is stored in "fyp_location.csv"
lat_val = loc_data["Lat"]
lon_val = loc_data["Lon"]
text_val = loc_data["Project Name"]
state_val = loc_data["State"]

numLocs = len(lat_val)
#numLocs = 10  #manually input 10 as not all data has been downloaded
#state = "VIC data/"

#Uncomment this for a variable energy requirement profile 

requiredMWh = []

# 24 hours per day, 365 days per year
for day in range(365):
    for hour in range(24):
        # Working hours: 8 <= hour < 17
        if 8<= hour <10:
            requiredMWh.append(25)
        elif 10 <= hour <18:
            requiredMWh.append(100)
        else:
            requiredMWh.append(0)
#data *starts* at 11 AM
requiredMWh = requiredMWh[11:] + requiredMWh[:11]
requiredMWh = np.array(requiredMWh)

in_yr = "2021"

if(True):
    
    solarCFs = []    #adds empty array to solarCFs
    windCFs = [] 
    
    for loc in range(numLocs):  #opens each location's file

        filename_solar = directory_main + str(state_val[loc]) + " data/" + in_yr + "_pv_" + str(lat_val[loc]) + "_" + str(lon_val[loc]) + ".csv"
        filename_wind = directory_main + str(state_val[loc]) + " data/" + in_yr + "_wind_" + str(lat_val[loc]) + "_" + str(lon_val[loc]) + ".csv"
   
        solarDataraw = pd.read_csv(str(filename_solar),skiprows=3)
        solarData = solarDataraw.to_xarray()

        #print(solarData)
     
        windDataraw = pd.read_csv(str(filename_wind),skiprows=3)
        windData = windDataraw.to_xarray()

        solarCFs.append(solarData["electricity"].values / 0.9) # nb we divide by 0.9 because the inversion losses are included in the Gen cost estimates
           #using .extend is better practice -> solarCFs now = [electricity data at location 1][electricity data at location 2][][][]
        windCFs.append(windData["electricity"].values / 0.9) 

      #cfs.append( data["specific generation"] )

    solarCFs = np.array(solarCFs)    #rewrite the solarCFs array
    windCFs = np.array(windCFs)
    
    #print(solarCFs.shape)     #prints out the shape of the array
    #print(windCFs.shape)
    
    #requiredMWh = 100*np.ones_like(solarCFs[0,:])
    print(requiredMWh.size)

    model_result = []
    
    for loc in range(numLocs):
        
        print("running optimization",flush=True)
        solarCFs_loop = solarCFs[[loc],:]
        windCFs_loop = windCFs[[loc],:]
        #print(windCFs_loop.shape)
        res = run_single_optimization(solarCFs_loop, windCFs_loop, requiredMWh)

        solarMWs  = res["solar_mw"] 
        windMWs = res["wind_mw"] 
        batteryMW = res["battery_mw"]
        annualized_cost = res["anualizedCost"]
        lcoe = annualized_cost/np.sum(requiredMWh)

        deficit,d_count,storage,totgen = CalculatePowerDeficit(solarCFs_loop, windCFs_loop, requiredMWh, solarMWs, windMWs, batteryMW)
        #np.savetxt(directory_main + "Results/" + text_val[loc] + "_deficit_2019.csv",deficit, delimiter = ',')  #saves the initial year deficit result in the Results folder

        plant_cost = PlantAnnualizedCostFunction(solarMWs, windMWs, batteryMW)
        penalty_cost = np.sum(deficit) * 20000

        model_result.append([text_val[loc],solarMWs[0],windMWs[0],batteryMW,annualized_cost,lcoe,plant_cost,penalty_cost])
        df = pd.DataFrame(model_result, columns = ["Location", "PV Size (MW)", "Wind Size (MW)", 
                                        "Battery Size (MW)", "Annualized Cost ($)", 
                                        "LCOE ($)", "Plant Cost ($)", "Model Year Penalty Cost ($)"])
        df.to_csv(directory_main + in_yr + "_Results/" + "Model Result Nelder.csv", index=False) 
        
        print("\n\n")
        print(text_val[loc])
        #print(f"solar MW: {solarMWs[0]:,.2f}")
        #print(f"wind MW: {windMWs[0]:,.2f}")
        #print(f"battery MW: {batteryMW:,.2f}")
        #print(f"Annualized Cost: ${annualized_cost:,.0f}")
        print( res["success"] )
        print( res["message"] )
        print( f"LCOE estimate: ${lcoe:,.2f}")
        print(f"The deficit count:{d_count}")

        #below is for trouble shooting 
        """
        # Print the first 24 hours of demand and solar CF to verify alignment
        print("Hour | Demand | Solar CF")
        for i in range(24):
            print(f"  {i:2d} | {requiredMWh[i]:6.0f} | {solarCFs_loop[0,i]:.3f}")
        print(f"Plant cost: ${plant_cost:,.0f}")
        print(f"Deficit penalty: ${penalty_cost:,.0f}")
        print(f"Total deficit MWh: {np.sum(deficit):.1f}")
        print(f"LCOE (plant cost only): ${plant_cost / np.sum(requiredMWh):.2f}/MWh")
        
        solar_capex = 352.80 * 1.414e6
        wind_capex = 29.92 * 3.113e6
        batt_capex = 170.59 * 1.652e6

        af25 = calculate_annuity_factor(25, 0.0599)
        af10 = calculate_annuity_factor(10, 0.0599)

        print(f"Solar annualized: ${af25 * solar_capex:,.0f}")
        print(f"Wind annualized: ${af25 * wind_capex:,.0f}")
        print(f"Battery annualized: ${af10 * batt_capex:,.0f}")
        """
        
        #pl.plot(deficit)
        #pl.title("Hours in the year vs Energy Deficit (2019)" + text_val[loc])
        #pl.xlabel("Hours in the year (h)")
        #pl.ylabel("Power Generation Deficit (MWh)")
        #pl.show()
  

        #code to run the data through 2020-2024
        data_years = [2019,2020,2021,2022,2023,2024] 
        deficit_future = []  

        solarCF_future = []  #adds empty array to solarCFs
        windCF_future = []

        for i in data_years:      
            filename_solar = directory_main + str(state_val[loc]) + " data/" + str(i) + "_pv_" + str(lat_val[loc]) + "_" + str(lon_val[loc]) + ".csv"
            filename_wind = directory_main + str(state_val[loc]) +  " data/" + str(i) + "_wind_" + str(lat_val[loc]) + "_" + str(lon_val[loc]) + ".csv"
        
            solarDataraw = pd.read_csv(str(filename_solar),skiprows=3)
            solarData = solarDataraw.to_xarray()
            
            windDataraw = pd.read_csv(str(filename_wind),skiprows=3)
            windData = windDataraw.to_xarray()

            solarData1 = solarData["electricity"].values / 0.9 # nb we divide by 0.9 because the inversion losses are included in the Gen cost estimates
            #using .extend is better practice -> solarCFs now = [electricity data at location 1][electricity data at location 2][][][]
            windData1 = windData["electricity"].values / 0.9
            
            solarData1 = solarData1[:8760]
            windData1 = windData1[:8760]

            solarCF_future.append(solarData1)
            windCF_future.append(windData1)

            
        #print("look look", windCF_future)
        solarCF_future = np.array(solarCF_future)    #rewrite the solarCFs array
        windCF_future = np.array(windCF_future)     
        #print(solarCF_future.shape)
        
        j=0
        all_rows = []
        row_labels = []
        for i in data_years:   
            solarCF_floop = solarCF_future[[j],:]
            windCF_floop = windCF_future[[j],:]
            #print(windCF_floop.shape)
            deficit_future1,d_count,storage,totgen = CalculatePowerDeficit(solarCF_floop, windCF_floop, requiredMWh, solarMWs, windMWs, batteryMW)
            #pl.plot(deficit_future1)
            #pl.xlabel("Hours in the year (h)")
            #pl.ylabel("Power Generation Deficit (MWh)")
            #pl.title(str(i))
            #pl.show()
            """
            deficit_future1 = np.concatenate([[d_count],deficit_future1])
            deficit_future.append(deficit_future1)
            genperhr = np.concatenate([[],totgen])
            deficit_future.append(genperhr)
            storage = np.concatenate([[666],storage])
            deficit_future.append(storage)
            """
            all_rows.append(deficit_future1)
            all_rows.append(totgen)
            all_rows.append(storage)

            row_labels.append(f"{i} deficit (MW)")
            row_labels.append(f"{i} generation (MW)")
            row_labels.append(f"{i} storage (MW)")

            #print("deficit count in year: " + str(i) + " is:" + str(d_count))
            j=j+1

        #deficit_future = np.transpose(deficit_future)
        results_df = pd.DataFrame(all_rows, index=row_labels).T
        #results_df.to_csv(directory_main + in_yr + "_Results/deficit_" + text_val[loc] + ".csv", index=False)
        #np.savetxt(directory_main + in_yr + "_Results/deficit_" + text_val[loc] + ".csv",deficit_future, delimiter = ',') #saves the all 5 future year results in Results folder for each location
    

else:
    # dummy data for testing
    solarCFs = np.random.rand(5,8760)   #creates radom number 1 x 58760 array
