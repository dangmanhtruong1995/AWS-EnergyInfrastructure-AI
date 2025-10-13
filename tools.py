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
from config import DATASET_LEGEND_DICT, DATASET_LIST, SCENARIOS
from utils import calculate_distance, load_data_and_process, add_data_source
from schemas import DataSourceTracker, GetWellEntryInput,\
    WellEntryOutput, DataSourceOutput, SeismicAndDrillingInput,\
    SeismicAndDrillingOutput, PlotOutput, SeismicAndLicensedBlocksInput,\
    AnalysisOutput, AvailableDataSources, ReportMapOutput
# from document_processor import extract_text_from_pdf
# from data_loader import get_coords
# from seismic_analysis_python import SeismicDrillingAnalyzer
from scenario_modeling import run_mcda, run_scenario_analysis



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

    add_data_source(run_context, layer_list)

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
       
    map_html = m.get_root().render()

    df_rank["Coordinates"] = df_rank["geometry"].centroid
    df_rank = df_rank.drop('geometry', axis=1)
    report = df_rank.to_string(index=False)
    report = "REPORT OF 20 MOST RELEVANT POINTS FOUND DURING THE MULTI-CRITERIA DECISION ANALYSIS, ALONG WITH SCORES (lower score is better) \n\n" + report
    
    # return report

    # Return structured data as JSON string

    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)


def within_op(layer_1: str, layer_2:str) -> gpd.GeoDataFrame:
    """ Perform a "within" operation, such as "Find all seismic events within licensed blocks".
    Args:
        layer_1: The first layer. For the example query "Find all seismic events within licensed blocks", the layer would be "seismic". The layer name should be chosen from the results of "get_available_data_sources".
        layer_2: The second layer. For the example query "Find all seismic events within licensed blocks", the layer would be "licences". The layer name should be chosen from the results of "get_available_data_sources".
    Return:
        df_rank: A GeoPandas's GeoDataFrame which lists the matched entries.
    """

    df_dict = {}
    df_dict[layer_1] = load_data_and_process(layer_1)
    df_dict[layer_2] = load_data_and_process(layer_2)
    
    if isinstance(df_dict[layer_1], pd.DataFrame):
        if 'Lon' in df_dict[layer_1].columns:
            geometry = [Point(xy) for xy in zip(df_dict[layer_1].Lon, df_dict[layer_1].Lat)]
            df_dict[layer_1] = df_dict[layer_1].drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_dict[layer_1]["geometry"]
        df_dict[layer_1] = gpd.GeoDataFrame(df_dict[layer_1], crs="EPSG:4326", geometry=geometry)

    if isinstance(df_dict[layer_2], pd.DataFrame):
        if 'Lon' in df_dict[layer_2].columns:
            geometry = [Point(xy) for xy in zip(df_dict[layer_2].Lon, df_dict[layer_2].Lat)]
            df_dict[layer_2] = df_dict[layer_2].drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_dict[layer_2]["geometry"]
        df_dict[layer_2] = gpd.GeoDataFrame(df_dict[layer_2], crs="EPSG:4326", geometry=geometry)

    try:
        utm_crs = df_dict[layer_1].estimate_utm_crs()
        df_dict[layer_1] = df_dict[layer_1].to_crs(utm_crs)
        df_dict[layer_2] = df_dict[layer_2].to_crs(utm_crs)
    except:
        pass

    n_within = np.zeros(len(df_dict[layer_1]))
    for idx in range(len(df_dict[layer_1])):
        row_geometry = df_dict[layer_1]["geometry"].iloc[idx]
        contains = row_geometry.contains(df_dict[layer_2]["geometry"])
        n_within[idx] = np.sum(contains)

    df_dict[layer_1]["Score"] = n_within.tolist()
    df_rank = df_dict[layer_1].sort_values("Score", ascending=False).copy()    
    df_rank = df_rank[df_rank["Score"] > 0]

    try:
        df_rank = df_rank.to_crs(epsg=4326)
    except:
        pass

    return df_rank


def within_dist_op(layer_1:str, layer_2:str, dist=10):
    """ Perform a "within distance" operation, such as "Find all licencing blocks which are within 10 kilometres of pipelines".
    Args:
        layer_1: The first layer. For the example query "Find all licencing blocks which are within 10 kilometres of pipelines", the layer would be "licences". The layer name should be chosen from the results of "get_available_data_sources".
        layer_2: The second layer. For the example query "Find all licencing blocks which are within 10 kilometres of pipelines", the layer would be "pipelines". The layer name should be chosen from the results of "get_available_data_sources".
    Return:
        df_rank: A GeoPandas's GeoDataFrame which lists the matched entries.
    """
    
    df_dict = {}
    df_dict[layer_1] = load_data_and_process(layer_1)
    df_dict[layer_2] = load_data_and_process(layer_2)

    if isinstance(df_dict[layer_1], pd.DataFrame):
        if 'Lon' in df_dict[layer_1].columns:
            geometry = [Point(xy) for xy in zip(df_dict[layer_1].Lon, df_dict[layer_1].Lat)]
            df_dict[layer_1] = df_dict[layer_1].drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_dict[layer_1]["geometry"]
        df_dict[layer_1] = gpd.GeoDataFrame(df_dict[layer_1], crs="EPSG:4326", geometry=geometry)

    if isinstance(df_dict[layer_2], pd.DataFrame):
        if 'Lon' in df_dict[layer_2].columns:
            geometry = [Point(xy) for xy in zip(df_dict[layer_2].Lon, df_dict[layer_2].Lat)]
            df_dict[layer_2] = df_dict[layer_2].drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_dict[layer_2]["geometry"]
        df_dict[layer_2] = gpd.GeoDataFrame(df_dict[layer_2], crs="EPSG:4326", geometry=geometry)

    try:
        utm_crs = df_dict[layer_1].estimate_utm_crs()
        df_dict[layer_1] = df_dict[layer_1].to_crs(utm_crs)
        df_dict[layer_2] = df_dict[layer_2].to_crs(utm_crs)
    except:
        pass
    
    # Nearest join (one match per row in df1)
    df_nearest = gpd.sjoin_nearest(
        df_dict[layer_1], df_dict[layer_2],
        how="left",
        distance_col="Score",
        max_distance=None  # set to dist if you want filtering here
    )

    # Convert meters → km
    #df_nearest["Score"] = df_nearest["Score"] / 1000.0
    df_nearest["Score"] = df_nearest["Score"] *1000.0
    
    # Align back to df1 index (handles duplicates safely)
    df_dict[layer_1]["Score"] = df_nearest.groupby(level=0)["Score"].first()
    df_rank = df_dict[layer_1][df_dict[layer_1]["Score"] <= dist].sort_values("Score")

    try:
        df_rank = df_rank.to_crs(epsg=4326)
    except:
        pass

    return df_rank


def analyse_and_plot_features_and_nearby_infrastructure(run_context: RunContext[DataSourceTracker], layer_1: str, layer_2: str, max_distance=10):
    """
    Analyse, then plot infrastructures in layer_1 that are close to those in layer_2
    
    Args:
        layer_1: layer containing point features (e.g., wells, facilities, stations). Layer names can be one of the available data source names (e.g. "wells").
        layer_2: layer containing linear infrastructure (e.g., pipelines, roads, cables). Layer names can be one of the available data source names (e.g. "pipelines").
        max_distance: maximum distance in km

    Return: A JSON containing the following:
        report: The final report in text form.
        map_html: The map HTML.
    """
    
    print(run_context)
    print()

    # Load the data. If the name doesn't match, try to search for closest match.
    try:
        df_points_ranked = within_dist_op(layer_1, layer_2, max_distance)
        df_lines = load_data_and_process(layer_2)
    except KeyError:
        dist_list = [nltk.edit_distance(layer_1, elem) for elem in DATASET_LIST]
        layer_1 = DATASET_LIST[np.argmin(dist_list)]

        dist_list = [nltk.edit_distance(layer_2, elem) for elem in DATASET_LIST]
        layer_2 = DATASET_LIST[np.argmin(dist_list)]

        df_points_ranked = within_dist_op(layer_1, layer_2, max_distance)
        df_lines = load_data_and_process(layer_2)

    if isinstance(df_lines, pd.DataFrame):
        if 'Lon' in df_lines.columns:
            geometry = [Point(xy) for xy in zip(df_lines.Lon, df_lines.Lat)]
            df_lines = df_lines.drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_lines["geometry"]
        df_lines = gpd.GeoDataFrame(df_lines, crs="EPSG:4326", geometry=geometry)
    
    print(f"Layer 1: {layer_1}")
    print(f"Layer 2: {layer_2}")
    print()

    add_data_source(run_context, [layer_1, layer_2])

    # Calculate the center point based on the data
    if len(df_points_ranked) > 0:
        # Get bounds of the point data
        bounds = df_points_ranked.bounds
        center_lat = (bounds.miny.min() + bounds.maxy.max()) / 2
        center_lon = (bounds.minx.min() + bounds.maxx.max()) / 2
        
        # Calculate appropriate zoom level based on data extent
        lat_range = bounds.maxy.max() - bounds.miny.min()
        lon_range = bounds.maxx.max() - bounds.minx.min()
        max_range = max(lat_range, lon_range)
        
        # Rough zoom level calculation (adjust as needed)
        if max_range > 10:
            zoom_level = 5
        elif max_range > 5:
            zoom_level = 6
        elif max_range > 2:
            zoom_level = 7
        elif max_range > 1:
            zoom_level = 8
        else:
            zoom_level = 9
    else:
        # Fallback to UK center if no data
        center_lat, center_lon = 55.3781, -1.4360
        zoom_level = 6
    
    # Create the map centered on UK
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        # location=[55.3781, -1.4360],  # UK center
        # zoom_start=6,
        tiles="cartodb positron",
        width='100%',
        height='600px'
    )
    
    # Find which line infrastructure is near the selected point features
    # We'll use a buffer around the points to find intersecting lines
    utm_crs = df_points_ranked.estimate_utm_crs()
    df_points_utm = df_points_ranked.to_crs(utm_crs)
    df_lines_utm = df_lines.to_crs(utm_crs)
    
    # Create buffer around point features (in meters)
    buffer_distance = max_distance * 1000  # convert km to meters
    points_buffered = df_points_utm.copy()
    points_buffered['geometry'] = points_buffered['geometry'].buffer(buffer_distance)
    
    # Find line infrastructure that intersects with the buffered points
    nearby_lines = gpd.sjoin(df_lines_utm, points_buffered, predicate='intersects')
    nearby_lines = nearby_lines.drop_duplicates(subset=['Name_left'])  # Remove duplicate lines
    nearby_lines = nearby_lines.to_crs(epsg=4326)
    
    # Convert point features back to WGS84 for plotting
    df_points_to_plot = df_points_ranked.to_crs(epsg=4326)
    
    # Plot the line infrastructure first (so they appear under the points)
    for index, row in nearby_lines.iterrows():
        try:
            geom = row["geometry"]
            popup_text = f"""
            <b>{DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</b><br>
            Name: {row.get('Name_left', 'Unknown')}<br>
            Type: Infrastructure
            """

            if geom.geom_type == 'LineString':
                # Single LineString
                coords = list(geom.coords)
                folium_coords = [[lat, lon] for lon, lat in coords]
                
                folium.PolyLine(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                    color='blue',
                    weight=3,
                    opacity=0.8
                ).add_to(m)
                
            elif geom.geom_type == 'MultiLineString':
                # Multiple LineString segments
                for line in geom.geoms:
                    coords = list(line.coords)
                    folium_coords = [[lat, lon] for lon, lat in coords]
                    
                    folium.PolyLine(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                        color='blue',
                        weight=3,
                        opacity=0.8
                    ).add_to(m)

            elif geom.geom_type == 'Polygon':
                # Extract exterior coordinates
                exterior_coords = list(geom.exterior.coords)
                folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                
                folium.Polygon(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                    color='blue',
                    weight=2,
                    opacity=0.8,
                    fillColor='lightblue',
                    fillOpacity=0.3
                ).add_to(m)
            
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    exterior_coords = list(polygon.exterior.coords)
                    folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                    
                    folium.Polygon(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                        color='blue',
                        weight=2,
                        opacity=0.8,
                        fillColor='lightblue',
                        fillOpacity=0.3
                    ).add_to(m)
            
            elif geom.geom_type == 'Point':
                lat = geom.y
                lon = geom.x
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(m)
                
        except Exception as e:
            print(f"Error plotting {layer_2} {index}: {e}")
            continue
    
    # Plot the point features
    points_added = 0
    for index, row in df_points_to_plot.iterrows():
        try:
            geom = row["geometry"]
            # set_trace()
            popup_text = f"""
            <b>{DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</b><br>
            Name: {row.get('Name_left', 'Unknown')}<br>
            Distance to {layer_2.title()}: {row['Score']:.3f} km<br> 
            Status: {row.get('ORIGINSTAT', 'N/A')}
            """
            
            # Handle both LineString and MultiLineString geometries
            if geom.geom_type == 'LineString':
                # Single LineString
                coords = list(geom.coords)
                folium_coords = [[lat, lon] for lon, lat in coords]
                
                folium.PolyLine(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_1.title()}: {row.get('Name_left', 'Unknown')}",
                    color='red',
                    weight=3,
                    opacity=0.8
                ).add_to(m)
                points_added += 1
                
            elif geom.geom_type == 'MultiLineString':
                # Multiple LineString segments
                for line in geom.geoms:
                    coords = list(line.coords)
                    folium_coords = [[lat, lon] for lon, lat in coords]
                    
                    folium.PolyLine(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_1.title()}: {row.get('Name_left', 'Unknown')}",
                        color='red',
                        weight=3,
                        opacity=0.8
                    ).add_to(m)
                points_added += 1

            # Handle Polygon geometries - NEW
            elif geom.geom_type == 'Polygon':
                # Extract exterior coordinates
                exterior_coords = list(geom.exterior.coords)
                folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                
                folium.Polygon(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_1.title()}: {row.get('Name_left', 'Unknown')}",
                    color='red',
                    weight=2,
                    opacity=0.8,
                    fillColor='lightblue',
                    fillOpacity=0.3
                ).add_to(m)
                points_added += 1
            
            # Handle MultiPolygon geometries - NEW
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    exterior_coords = list(polygon.exterior.coords)
                    folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                    
                    folium.Polygon(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_1.title()}: {row.get('Name_left', 'Unknown')}",
                        color='red',
                        weight=2,
                        opacity=0.8,
                        fillColor='lightblue',
                        fillOpacity=0.3
                    ).add_to(m)
                points_added += 1
            
            # Handle Point geometries (in case layer_2 contains points) - NEW
            elif geom.geom_type == 'Point':
                lat = geom.y
                lon = geom.x
                
                # if -90 <= lat <= 90 and -180 <= lon <= 180:
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_1.title()}: {row.get('Name_left', 'Unknown')}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                points_added += 1
                
        except Exception as e:
            print(f"Error plotting {layer_1} {index}: {e}")
            continue   
    
    print(f"Successfully added {points_added} {layer_1} and {len(nearby_lines)} {layer_2} to the map")
    
    # Enhanced legend
    row = df_points_ranked.iloc[0, :]
    geom = row["geometry"]
    if geom.geom_type == 'LineString':
        legend_text_1 = f'<p><span style="color:red; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</p>'                
    elif geom.geom_type == 'MultiLineString':
        legend_text_1 = f'<p><span style="color:red; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</p>'
    elif geom.geom_type == 'Polygon':
        legend_text_1 = f'<span style="color:red; border: 1px solid blue; background-color:lightblue; padding:2px 6px; display:inline-block;">▭ {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</span>'
    elif geom.geom_type == 'MultiPolygon':
        legend_text_1 = f'<span style="color:red; border: 1px solid blue; background-color:lightblue; padding:2px 6px; display:inline-block;">▭ {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</span>'
    elif geom.geom_type == 'Point':
        legend_text_1 = f"<p><i class='fa fa-map-marker' style='color:red'></i> {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</p>"

    row = df_lines.iloc[0, :]
    geom = row["geometry"]
    if geom.geom_type == 'LineString':
        legend_text_2 = f'<p><span style="color:blue; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>'                
    elif geom.geom_type == 'MultiLineString':
        legend_text_2 = f'<p><span style="color:blue; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>'
    elif geom.geom_type == 'Polygon':
        legend_text_2 = f'<span style="color:blue; border: 1px solid blue; background-color:lightblue; padding:2px 6px; display:inline-block;">▭ {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</span>'
    elif geom.geom_type == 'MultiPolygon':
        legend_text_2 = f'<span style="color:blue; border: 1px solid blue; background-color:lightblue; padding:2px 6px; display:inline-block;">▭ {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</span>'
    elif geom.geom_type == 'Point':
        legend_text_2 = f"<p><i class='fa fa-map-marker' style='color:blue'></i> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>"
   
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legend</h4>
    {legend_text_1}
    {legend_text_2}
    </div>
    '''
    """
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legend</h4>
    <p><i class="fa fa-map-marker" style="color:red"></i> {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</p>
    <p><span style="color:blue; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>    
    </div>
    '''
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    map_html = m.get_root().render()

    # Generate report
    df_points_ranked["Coordinates"] = df_points_ranked["geometry"].centroid
    df_points_ranked = df_points_ranked.drop('geometry', axis=1)
    try:
        df_points_ranked = df_points_ranked.drop('ORIGINSTAT', axis=1)
    except:
        pass
    report = df_points_ranked.to_string(index=False)
    report = f"REPORT of {layer_1} assets which are close to {max_distance} kilometres of {layer_2} assets : \n\n" + report
    
    print(report)
    print()

    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)


def analyse_using_mcda_then_plot(run_context: RunContext, target:str,
        obj_1: str, obj_2: str, obj_3: str, obj_4: str,
        w_1: float, w_2: float, w_3: float, w_4: float):    
    """ Do a Multi-Criterion Decision Analysis (MCDA) for the given target,
        ranking by a number of given objectives. After that, plot the results.
    Args:
        target: The target for analysis (e.g. "licence")
        obj_1: The 1st objective (e.g. "safety")
        obj_2: The 2nd objective (e.g. "environment")
        obj_3: The 3rd objective (e.g. "technical")
        obj_4: The 4th objective (e.g. "economic")
        w_1: The weight for the 1st objective
        w_2: The weight for the 2nd objective
        w_3: The weight for the 3rd objective
        w_4: The weight for the 4rd objective
    Return: A JSON containing the following:
        report: The final report in text form. Please note that for the objective scores, lower is better.
        map_html: The HTML content showing the map.
    """

    print(run_context)
    print()

    layer_list = ["licences", "wells", "seismic", "drilling", "pipelines", "offshore_fields"]
    add_data_source(run_context, layer_list)

    report, df_rank = run_mcda(target, obj_1, obj_2, obj_3, obj_4, w_1, w_2, w_3, w_4)
    df_rank = df_rank.rename(columns={'Coordinates': 'geometry'})
    df_rank.set_geometry("geometry")
    
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
            fields=["Name", "Score", "Rank", "safety_score", "environment_score", "technical_score", "economic_score"],
            aliases=["Licence", "Total Score", "Rank", "Safety", "Environment", "Technical", "Economic"],
            localize=True
        ),
    ).add_to(m)
    
    # Custom JavaScript for cluster icon that shows average rank
    with open('templates/mcda_cluster_icons.jstemplate', 'r') as file:
        cluster_icon_js = file.read()
   
    # Create cluster with custom icon function
    cluster = MarkerCluster(
        icon_create_function=cluster_icon_js,
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
       
    map_html = m.get_root().render()

    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)


def get_scenario_weights(run_context: RunContext, scenario_name: str = None) -> str:
    """
    Get the weight configuration for available MCDA scenarios.
    
    Args:
        scenario_name: Optional specific scenario name to query. If None, returns all scenarios.
                      Valid values: "balanced", "economic_focus", "safety_focus", "technical_focus", "environment_focus"
    
    Return:
        JSON string containing scenario weights information
    """

    layer_list = ["licences", "wells", "seismic", "drilling", "pipelines", "offshore_fields"]
    add_data_source(run_context, layer_list)

    if scenario_name is None:
        # Return all scenarios
        result = {
            "available_scenarios": list(SCENARIOS.keys()),
            "all_weights": SCENARIOS
        }
        return json.dumps(result, indent=2)
    
    # Return specific scenario
    if scenario_name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        return json.dumps({
            "error": f"Scenario '{scenario_name}' not found",
            "available_scenarios": available
        })
    
    result = {
        "scenario": scenario_name,
        "weights": SCENARIOS[scenario_name],
        "description": f"In {scenario_name}, the weights are: " + 
                      ", ".join([f"{k}={v}" for k, v in SCENARIOS[scenario_name].items()])
    }
    
    return json.dumps(result, indent=2)


def perform_scenario_analysis_then_plot(run_context: RunContext,
        scenario_name: str,
        adjust_safety: float = 0.0, 
        adjust_technical: float = 0.0, 
        adjust_economic: float = 0.0, 
        adjust_environment: float = 0.0):
    """
    Run scenario analysis using Multi-Criterion Decision Analysis (MCDA), then plot the results.
    
    Available scenarios:
    - "balanced": All objectives weighted equally at 0.25
    - "economic_focus": Economic weighted at 0.5, others at 0.1-0.2
    - "safety_focus": Safety weighted at 0.5, others at 0.1-0.2
    - "technical_focus": Technical weighted at 0.5, others at 0.1-0.2
    - "environment_focus": Environment weighted at 0.5, others at 0.1-0.2
    
    Args:
        scenario_name: Name of the base scenario (e.g., 'safety_focus')
        adjust_safety: Adjustment to safety weight (e.g., +0.1 to increase by 0.1, -0.1 to decrease)
        adjust_technical: Adjustment to technical weight (e.g., +0.2 to increase by 0.2)
        adjust_economic: Adjustment to economic weight
        adjust_environment: Adjustment to environment weight
    
    Examples:
        - "Run safety_focus with technical doubled" → scenario_name="safety_focus", adjust_technical=0.2
        - "Run balanced scenario with more focus on environment" → scenario_name="balanced", adjust_environment=0.15

    Return: A JSON containing:
        report: The report in text form with rankings
        map_html: Interactive map visualization
    """

    print(run_context)
    print()

    

    # Build adjust dict from parameters
    adjust = {}
    if adjust_safety != 0.0:
        adjust['safety'] = adjust_safety
    if adjust_technical != 0.0:
        adjust['technical'] = adjust_technical
    if adjust_economic != 0.0:
        adjust['economic'] = adjust_economic
    if adjust_environment != 0.0:
        adjust['environment'] = adjust_environment
    
    adjust = adjust if adjust else None

    report, df_rank, used_weights = run_scenario_analysis(scenario_name, adjust=adjust)
    report += f"\n USED WEIGHTS: {used_weights}."

    df_rank = df_rank.rename(columns={'Coordinates': 'geometry'})
    df_rank.set_geometry("geometry")
    
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
            fields=["Name", "Score", "Rank", "safety_score", "environment_score", "technical_score", "economic_score"],
            aliases=["Licence", "Total Score", "Rank", "Safety", "Environment", "Technical", "Economic"],
            localize=True
        ),
    ).add_to(m)
    
    # Custom JavaScript for cluster icon that shows average rank
    with open('templates/mcda_cluster_icons.jstemplate', 'r') as file:
        cluster_icon_js = file.read()
   
    # Create cluster with custom icon function
    cluster = MarkerCluster(
        icon_create_function=cluster_icon_js,
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

    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)


def analyse_and_plot_within_op(run_context: RunContext[DataSourceTracker], layer_1: str, layer_2: str):
    """
    Analyse, then plot features in layer_1 that are within (contained by) features in layer_2.
    
    Args:
        layer_1: The layer containing features to check if they fall within layer_2 (e.g., "wells", "seismic", "drilling"). 
                 These are typically point features or smaller geometries.
        layer_2: The layer containing container features (e.g., "licences", "offshore_fields"). 
                 These are typically polygon features that can contain layer_1 features.
    
    Examples:
        - "Find all wells which are within licence blocks" → layer_1="wells", layer_2="licences"        
        - "Find drilling locations within licensed blocks" → layer_1="drilling", layer_2="licences"
        - "Show wells within offshore fields" → layer_1="wells", layer_2="offshore_fields"


    Return: A JSON containing the following:
        report: The final report in text form.
        map_html: The map HTML.
    """
    print(run_context)
    print()

    print(f"Layer 1: {layer_1}")
    print(f"Layer 2: {layer_2}")

    # Load the data. If the name doesn't match, try to search for closest match.
    if layer_1 not in DATASET_LIST:
        dist_list = [nltk.edit_distance(layer_1, elem) for elem in DATASET_LIST]
        layer_1 = DATASET_LIST[np.argmin(dist_list)]

    if layer_2 not in DATASET_LIST:
        dist_list = [nltk.edit_distance(layer_2, elem) for elem in DATASET_LIST]
        layer_2 = DATASET_LIST[np.argmin(dist_list)]

    df_rank = within_op(layer_1, layer_2)
    if len(df_rank) == 0:
        temp = layer_1
        layer_1 = layer_2
        layer_2 = temp

        df_rank = within_op(layer_1, layer_2)
    df_layer2 = load_data_and_process(layer_2)

    print("df_rank")
    print(df_rank)

    if isinstance(df_layer2, pd.DataFrame):
        if 'Lon' in df_layer2.columns:
            geometry = [Point(xy) for xy in zip(df_layer2.Lon, df_layer2.Lat)]
            df_layer2 = df_layer2.drop(['Lon', 'Lat'], axis=1)
        else:
            geometry = df_layer2["geometry"]
        df_layer2 = gpd.GeoDataFrame(df_layer2, crs="EPSG:4326", geometry=geometry)
    
    print(f"Layer 1: {layer_1}")
    print(f"Layer 2: {layer_2}")
    print()

    # Add data sources to tracker
    add_data_source(run_context, [layer_1, layer_2])

    # Calculate center and zoom
    if len(df_rank) > 0:
        bounds = df_rank.bounds
        center_lat = (bounds.miny.min() + bounds.maxy.max()) / 2
        center_lon = (bounds.minx.min() + bounds.maxx.max()) / 2
        
        lat_range = bounds.maxy.max() - bounds.miny.min()
        lon_range = bounds.maxx.max() - bounds.minx.min()
        max_range = max(lat_range, lon_range)
        
        if max_range > 10:
            zoom_level = 5
        elif max_range > 5:
            zoom_level = 6
        elif max_range > 2:
            zoom_level = 7
        elif max_range > 1:
            zoom_level = 8
        else:
            zoom_level = 9
    else:
        center_lat, center_lon = 55.3781, -1.4360
        zoom_level = 6
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles="cartodb positron",
        width='100%',
        height='600px'
    )
    
    # Convert to WGS84 for plotting
    df_rank_plot = df_rank.to_crs(epsg=4326)
    df_layer2_plot = df_layer2.to_crs(epsg=4326)
    
    # Find which layer_2 features are within layer_1 features
    # utm_crs = df_rank.estimate_utm_crs()
    utm_crs = df_rank.to_crs(epsg=4326).estimate_utm_crs()
    df_rank_utm = df_rank.to_crs(utm_crs)
    df_layer2_utm = df_layer2.to_crs(utm_crs)
    
    # Spatial join to find contained features
    contained_layer2 = gpd.sjoin(df_layer2_utm, df_rank_utm, predicate='within')
    contained_layer2 = contained_layer2.to_crs(epsg=4326)
    
    # Plot layer_1 polygons first (containers)
    for index, row in df_rank_plot.iterrows():
        try:
            geom = row["geometry"]
            popup_text = f"""
            <b>{DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</b><br>
            Name: {row.get('Name', 'Unknown')}<br>
            Contains {int(row['Score'])} {layer_2}
            """
            
            if geom.geom_type == 'Polygon':
                exterior_coords = list(geom.exterior.coords)
                folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                
                folium.Polygon(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_1.title()}: {row.get('Name', 'Unknown')}",
                    color='blue',
                    weight=2,
                    opacity=0.8,
                    fillColor='yellow',
                    fillOpacity=0.4
                ).add_to(m)
            
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    exterior_coords = list(polygon.exterior.coords)
                    folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                    
                    folium.Polygon(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_1.title()}: {row.get('Name', 'Unknown')}",
                        color='blue',
                        weight=2,
                        opacity=0.8,
                        fillColor='yellow',
                        fillOpacity=0.4
                    ).add_to(m)
        except Exception as e:
            print(f"Error plotting {layer_1} {index}: {e}")
            continue
    
    # Plot layer_2 features that are contained
    points_added = 0
    for index, row in contained_layer2.iterrows():
        try:
            geom = row["geometry"]
            popup_text = f"""
            <b>{DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</b><br>
            Name: {row.get('Name_left', 'Unknown')}<br>
            Within: {row.get('Name_right', 'Unknown')}
            """
            
            if geom.geom_type == 'Point':
                lat = geom.y
                lon = geom.x
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                points_added += 1
                
            elif geom.geom_type == 'LineString':
                coords = list(geom.coords)
                folium_coords = [[lat, lon] for lon, lat in coords]
                
                folium.PolyLine(
                    locations=folium_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                    color='red',
                    weight=3,
                    opacity=0.8
                ).add_to(m)
                points_added += 1
                
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    folium_coords = [[lat, lon] for lon, lat in coords]
                    
                    folium.PolyLine(
                        locations=folium_coords,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{layer_2.title()}: {row.get('Name_left', 'Unknown')}",
                        color='red',
                        weight=3,
                        opacity=0.8
                    ).add_to(m)
                points_added += 1
                
        except Exception as e:
            print(f"Error plotting {layer_2} {index}: {e}")
            continue
    
    print(f"Successfully added {len(df_rank_plot)} {layer_1} and {points_added} {layer_2} to the map")
    
    # Dynamic legend based on geometry types
    row = df_rank_plot.iloc[0, :]
    geom = row["geometry"]
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        legend_text_1 = f'<span style="color:blue; border: 1px solid blue; padding:2px 6px; display:inline-block;">▭ {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())}</span>'
    
    row = contained_layer2.iloc[0, :]
    geom = row["geometry"]
    if geom.geom_type == 'Point':
        legend_text_2 = f"<p><i class='fa fa-map-marker' style='color:red'></i> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>"
    elif geom.geom_type in ['LineString', 'MultiLineString']:
        legend_text_2 = f'<p><span style="color:red; font-weight:bold;">━━</span> {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}</p>'
    
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legend</h4>
    {legend_text_1}
    {legend_text_2}
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    map_html = m.get_root().render()
    # map_html = m._repr_html_()
    # with open("within_operation_plot.html", "w", encoding="utf-8") as f:
    #     f.write(map_html)
    
    # Generate report
    df_rank["Coordinates"] = df_rank["geometry"].centroid
    df_rank = df_rank.drop('geometry', axis=1)
    report = df_rank.to_string(index=False)
    report = f"REPORT of {layer_1} features that contain {layer_2} features:\n\n" + report
    
    # return report

    result = {
        'report': report,
        'map_html': map_html
    }
    
    return json.dumps(result)