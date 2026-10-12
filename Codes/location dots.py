import zipfile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

user_directory = "C:/Users/joyli/OneDrive/Desktop/FYP/Coding and Data/FYP_Hubs-or-Dubs/"

with zipfile.ZipFile("indicative-rez-boundaries-2026-gis-data.kmz", "r") as z:
    z.extractall("kmz_extracted")

loc = pd.read_csv(user_directory + "fyp_location.csv", encoding="latin1")
loc["Project Name"] = loc["Project Name"].str.strip()

projects = gpd.GeoDataFrame(
    loc,
    geometry=[Point(xy) for xy in zip(loc["Lon"], loc["Lat"])],
    crs="EPSG:4326"
)

rez = gpd.read_file("kmz_extracted/doc.kml").to_crs("EPSG:4326")

aus_map = gpd.read_file(user_directory + "SA2_2026_AUST_SHP_GDA2020/SA2_2026_AUST_GDA2020.shp").to_crs("EPSG:4326")
states_to_show = ["New South Wales", "Victoria", "Queensland", "South Australia", "Tasmania"]
aus_subset = aus_map[aus_map["STE_NAME26"].isin(states_to_show)]

fig, ax = plt.subplots(figsize=(10, 12))

aus_subset.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)
rez.plot(ax=ax, color="lightblue", edgecolor="black", linewidth=0.5, alpha=0.6)
projects.plot(ax=ax, color="red", markersize=15, zorder=5, marker="o", edgecolor="darkred", linewidth=0.5)

rez_patch = mpatches.Patch(color="lightblue", label="Renewable Energy Zones")
point_patch = mpatches.Patch(color="red", label="Project locations")
ax.legend(handles=[rez_patch, point_patch], loc="lower left", fontsize=8)

ax.set_axis_off()
#ax.set_title("Renewable Energy Zones and Project Locations")
plt.savefig("rez_map_1.png", dpi=300, bbox_inches="tight")
plt.show()