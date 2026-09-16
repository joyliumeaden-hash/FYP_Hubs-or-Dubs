import geopandas as gpd

user_directory = "C:/Users/joyli/OneDrive/Desktop/FYP/Coding and Data/"
directory = "FYP_Hubs-or-Dubs/SA2_2026_AUST_SHP_GDA2020/"
sa2 = gpd.read_file(user_directory + directory + "SA2_2026_AUST_GDA2020.shp")
#print(sa2.columns.tolist())
sa2.drop(columns="geometry").to_csv("sa2_regions.csv", index=False)