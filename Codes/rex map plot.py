import zipfile

user_directory = "C:/Users/joyli/OneDrive/Desktop/FYP/Coding and Data/FYP_Hubs-or-Dubs/"
result_directory = "C:/Users/joyli/OneDrive/Desktop/FYP/Results/"

with zipfile.ZipFile("indicative-rez-boundaries-2026-gis-data.kmz", "r") as z:
    print(z.namelist())  # see what's inside, usually "doc.kml"
    z.extractall("kmz_extracted")


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

loc = pd.read_csv(user_directory + "fyp_location.csv", encoding="latin1")

projects = gpd.GeoDataFrame(
    loc,
    geometry=[Point(xy) for xy in zip(loc["Lon"], loc["Lat"])],
    crs="EPSG:4326"
)





results = pd.read_csv(result_directory + "Model Result.csv")
results["Location"] = results["Location"].str.strip()

merged = loc.merge(results[["Location", "Plant Cost ($)"]],
                    left_on="Project Name", right_on="Location", how="left")

projects = gpd.GeoDataFrame(
    merged,
    geometry=[Point(xy) for xy in zip(merged["Lon"], merged["Lat"])],
    crs="EPSG:4326"
)

rez = gpd.read_file("kmz_extracted/doc.kml").to_crs("EPSG:4326")
matched = gpd.sjoin(projects, rez, how="left", predicate="within")

#result = matched[["Project Name", "Lat", "Lon", "State", "Name"]].rename(columns={"Name": "REZ"})

#result.to_csv("projects_matched_to_rez.csv", index=False)
#print(result)
aus_map = gpd.read_file(user_directory + "SA2_2026_AUST_SHP_GDA2020/SA2_2026_AUST_GDA2020.shp").to_crs("EPSG:4326")
states_to_show = ["New South Wales", "Victoria","Queensland","South Australia","Tasmania"]
rez_totals = matched.groupby("Name")["Plant Cost ($)"].sum().reset_index()
rez_totals = rez_totals.rename(columns={"Name": "REZ_Name"})
rez_plot = rez.merge(rez_totals, left_on="Name", right_on="REZ_Name", how="left")
aus_subset = aus_map[aus_map["STE_NAME26"].isin(states_to_show)]

fig, ax = plt.subplots(figsize=(10, 12))

aus_subset.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)
plot = rez_plot.plot(column="Plant Cost ($)", cmap="OrRd", legend=True,
              edgecolor="black", linewidth=0.5, ax=ax,
              missing_kwds={"color": "lightgrey", "label": "No projects"})

cbar = plot.get_figure().axes[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))

ax.set_axis_off()
ax.set_title("Total Plant Cost by Renewable Energy Zone In 2019 Using 2019 As Base Year")
plt.savefig("plant_cost_by_rez.png", dpi=300, bbox_inches="tight")
plt.show()