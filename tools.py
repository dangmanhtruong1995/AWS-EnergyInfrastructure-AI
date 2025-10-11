from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Set, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import json
# import contextily as ctx
import nltk
import sys
import time
from pdb import set_trace

from pyproj import Transformer
from geopy.geocoders import Nominatim
import utm
from shapely.geometry import Point
import geopandas as gpd
import geodatasets
import folium
from folium import plugins
import branca
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import folium
import matplotlib.colors as mcolors
from folium.plugins import MarkerCluster, BeautifyIcon

import pydantic_ai
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ToolReturnPart

import os 
from os.path import join as pjoin
import boto3

# from config import BASE_PATH, DATASETS, DATASET_LIST, DATASET_LEGEND_DICT
from utils import calculate_distance, load_data_and_process
# from schemas import DataSourceTracker, GetWellEntryInput,\
#     WellEntryOutput, DataSourceOutput, SeismicAndDrillingInput,\
#     SeismicAndDrillingOutput, PlotOutput, SeismicAndLicensedBlocksInput,\
#     AnalysisOutput, AvailableDataSources
# from document_processor import extract_text_from_pdf
# from data_loader import get_coords
# from seismic_analysis_python import SeismicDrillingAnalyzer
# from scenario_modeling import run_mcda, run_scenario_analysis



def mcda(run_context: RunContext, target:str, obj_1: str, obj_2: str, obj_3: str, obj_4):
    """ Do a Multi-Criterion Decision Analysis (MCDA) for the given target, ranking by a number of given objectives
    Args:
        target: The target for analysis (e.g. "licence")
        obj_1: The 1st objective (e.g. "safety")
        obj_2: The 2nd objective (e.g. "environment")
        obj_3: The 3rd objective (e.g. "technical")
        obj_4: The 4th objective (e.g. "safety")
    Return:
        report: The final report in text form. Please note that for the objective scores, lower is better.
    """

    print(run_context)
    print()

    obj_list = [obj_1, obj_2, obj_3, obj_4]
    print(f"TARGET: {target}")
    if ("afety" in obj_list) and ("echincal" in obj_list) and ("conomi" in obj_list) and ("nviron" in obj_list):
        print("Objectives ARE: Safety, technical, economic and environment")
    else:
        print(f"Objectives ARE: {obj_list}")

    df_dict = {}
    layer_list = ["licences", "wells", "seismic", "drilling", "pipelines", "offshore_fields"]

    for layer_name in layer_list:
        start = time.time()
        print(f"Loading {layer_name}...", flush=True)
        sys.stdout.flush()

        df_dict[layer_name] = load_data_and_process(layer_name)

        elapsed = time.time() - start
        print(f"✓ Loaded {layer_name} in {elapsed:.2f}s ({len(df_dict[layer_name])} rows)", flush=True)
        sys.stdout.flush()

    print("All data loaded. Starting analysis...", flush=True)
    sys.stdout.flush()

    n_licence = len(df_dict["licences"])
    n_well = len(df_dict["wells"])
   
    # 1) Safety
    print("Calculating safety objective")
    num_wells_within_licence = np.zeros(n_licence)
    num_old_wells_within_licence = np.zeros(n_licence)
    for licence_idx in range(n_licence):
        licence = df_dict["licences"]["geometry"].iloc[licence_idx]
        contains = licence.contains(df_dict["wells"]["geometry"])
        num_wells_within_licence[licence_idx] = np.sum(contains)
    for licence_idx in range(n_licence):
        licence = df_dict["licences"]["geometry"].iloc[licence_idx]
        contains = 0
        for well_idx in range(n_well):
            if (df_dict["wells"]['ORIGINSTAT'][well_idx] == 'Decommissioned') and (licence.contains(df_dict["wells"]["geometry"][well_idx])):
                contains += 1
        num_old_wells_within_licence[licence_idx] = contains  
    safety_obj = num_wells_within_licence + num_old_wells_within_licence

    # 2) Environment
    print("Calculating environment objective")
    num_seismic_within_licence = np.zeros(n_licence)
    for licence_idx in range(n_licence):
        licence = df_dict["licences"]["geometry"].iloc[licence_idx]
        contains = licence.contains(df_dict["seismic"]["geometry"])
        num_seismic_within_licence[licence_idx] = np.sum(contains)
    env_obj = num_seismic_within_licence

    # 3) Technical
    print("Calculating technical objective")
    num_pipelines_going_through_licence = np.zeros(n_licence)
    for licence_idx in range(n_licence):
        licence = df_dict["licences"]["geometry"].iloc[licence_idx]
        intersects = licence.intersects(df_dict["pipelines"]["geometry"])
        num_pipelines_going_through_licence[licence_idx] = np.sum(intersects)
    tech_obj = num_pipelines_going_through_licence
    
    # 4) Economic
    print("Calculating economic objective")
    dist_to_nearest_field_km = np.zeros(n_licence)
    for licence_idx in range(n_licence):
        licence = df_dict["licences"]["geometry"].iloc[licence_idx]
        dist_min = np.min(licence.distance(df_dict["offshore_fields"]["geometry"]))
        dist_to_nearest_field_km[licence_idx] = dist_min
    econ_obj = dist_to_nearest_field_km

    # Normalize and invert for technical (make all objectives higher is worse)
    safety_obj_normalized = (safety_obj - np.min(safety_obj)) / (np.max(safety_obj) - np.min(safety_obj))
    env_obj_normalized = (env_obj - np.min(env_obj)) / (np.max(env_obj) - np.min(env_obj))
    tech_obj_normalized = (tech_obj - np.min(tech_obj)) / (np.max(tech_obj) - np.min(tech_obj))
    tech_obj_normalized = 1 - tech_obj_normalized
    econ_obj_normalized = (econ_obj - np.min(econ_obj)) / (np.max(econ_obj) - np.min(econ_obj))

    # Multi-objective score
    w_HS = 0.9
    w_EN = 0.9
    w_TE = 0.1
    w_EC = 0.1
    
    score = w_HS*safety_obj_normalized + w_EN*env_obj_normalized + w_TE*tech_obj_normalized + w_EC*econ_obj_normalized

    df_licenses = df_dict["licences"].copy()
    df_licenses["Score"] = score.tolist()
    df_licenses["Safety_score"] = safety_obj_normalized.tolist()
    df_licenses["Environment_score"] = env_obj_normalized.tolist()
    df_licenses["Techincal_score"] = tech_obj_normalized.tolist()
    df_licenses["Economic_score"] = econ_obj_normalized.tolist()

    # Rank the licence blocks by score (higher is worse)
    df_rank = df_licenses.sort_values("Score").iloc[:20,:].copy()
    df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))
    
    # --- 1. Center map somewhere in UKCS ---
    # Use the centroid of all licence polygons
    m_center = df_rank.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[m_center.y, m_center.x], zoom_start=5, tiles="CartoDB positron")
    
    # --- 2. Define a color function (low = dark green, high = light yellow) ---
    def get_color(value):
        # value is normalized 0..1, we invert so low score = strong color        
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
        rgba = cmap(1 - value)
        return mcolors.to_hex(rgba)
    
    def style_function(feature):
        score = feature["properties"]["Score"]        
        norm_value = score
        return {
            "fillColor": get_color(norm_value),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.99,
        }
    
    # --- 3. Add licence polygons, color by overall Score ---
    folium.GeoJson(
        df_rank,
        style_function=style_function,       
        tooltip=folium.GeoJsonTooltip(            
            fields=["Name", "Score", "Rank", "Safety_score", "Environment_score", "Techincal_score", "Economic_score"],
            aliases=["Licence", "Total Score", "Rank", "Safety", "Environment", "Technical", "Economic"],
            localize=True
        ),
    ).add_to(m)
    
    # Custom JavaScript for cluster icon that shows average rank
    # with open('templates/mcda_cluster_icons.jstemplate', 'r') as file:
    #     cluster_icon_js = file.read()
   
    # Create cluster with custom icon function
    cluster = MarkerCluster(
    #     icon_create_function=cluster_icon_js,
    ).add_to(m)
    
    # Add markers to cluster
    for _, row in df_rank.iterrows():
        centroid = row.geometry.centroid
        # Here we use the rank itself as color intensity (you could use sum/mean if grouping)
        color = get_color(1 / row["Rank"])  # inverse rank: rank=1 is strongest green
    
        marker = folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"<b>Licence:</b> {row['Name']}<br><b>Score:</b> {row['Score']:.3f}<br><b>Rank:</b> {row['Rank']}",
            icon=BeautifyIcon(
                icon_shape="marker",
                border_color=color,
                background_color=color,
                text_color="white",
                number=row["Rank"],  # optional: show rank number inside cluster marker
            )
        )
    
        # Add rank data to marker options for cluster calculation
        marker.options['rank'] = int(row["Rank"])
        marker.add_to(cluster)
       
    # --- 3. Save to HTML ---
    # m.save("licence_scores_map.html")
    # map_html = m._repr_html_()
    # with open("licence_scores_map.html", "w", encoding="utf-8") as f:
    #     f.write(map_html)
    # print("✅ Saved interactive map as licence_scores_map.html")
    map_html = m.get_root().render()

    df_rank["Coordinates"] = df_rank["geometry"].centroid
    df_rank = df_rank.drop('geometry', axis=1)
    report = df_rank.to_string(index=False)
    report = "REPORT OF 20 MOST RELEVANT POINTS FOUND DURING THE MULTI-CRITERIA DECISION ANALYSIS, ALONG WITH SCORES (lower score is better) \n\n" + report
    
    # return report

    # Return structured data as JSON string
    import json
    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)