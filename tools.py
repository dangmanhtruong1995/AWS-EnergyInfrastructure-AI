import os 
from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Set, Union, Dict, List
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
from datetime import datetime, timedelta

from pyproj import Transformer
from geopy.geocoders import Nominatim
import utm
import geopandas as gpd
import geodatasets
import folium
from folium import plugins
import branca
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import folium
import matplotlib.colors as mcolors
from folium.plugins import MarkerCluster, BeautifyIcon, HeatMap

import pydantic_ai
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ToolReturnPart

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
from grid_system import ExplorationGridSystem
from global_wind_farm_planner import GlobalWindFarmPlanner,\
    create_wind_farm_map, generate_wind_farm_report
from get_location_region_bounds import _try_nominatim_with_boundingbox,\
    _try_nominatim_point_based, _get_bounds_from_coordinates,\
    _try_maritime_regions, _calculate_bounds_from_point

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
    if layer_1 not in DATASET_LIST:
        dist_list = [nltk.edit_distance(layer_1, elem) for elem in DATASET_LIST]
        layer_1 = DATASET_LIST[np.argmin(dist_list)]

    if layer_2 not in DATASET_LIST:
        dist_list = [nltk.edit_distance(layer_2, elem) for elem in DATASET_LIST]
        layer_2 = DATASET_LIST[np.argmin(dist_list)]

    df_points_ranked = within_dist_op(layer_1, layer_2, max_distance)
    if len(df_points_ranked) == 0:
        temp = layer_1
        layer_1 = layer_2
        layer_2 = temp
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

    # layer_list = ["licences", "wells", "seismic", "drilling", "pipelines", "offshore_fields"]
    layer_list = DATASET_LIST
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

    # layer_list = ["licences", "wells", "seismic", "drilling", "pipelines", "offshore_fields"]
    layer_list = DATASET_LIST
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


# === Tools relating to risk assessment ===
def geocode_location(run_context: RunContext[DataSourceTracker], location: str) -> str:
    """
    Convert a location name or address to latitude and longitude coordinates.
    
    Args:
        location: Location name, address, or coordinates (e.g., "Aberdeen, UK", "North Sea", "57.1497, -2.0943")
    
    Return:
        JSON string with coordinates and location details
    """
    
    try:
        
        
        # Try to parse if it's already coordinates
        if ',' in location:
            parts = location.split(',')
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        result = {
                            'latitude': lat,
                            'longitude': lon,
                            'location_name': f"Coordinates: {lat}, {lon}",
                            'success': True
                        }
                        return json.dumps(result)
                except ValueError:
                    pass
        
        # Use geocoding service
        geolocator = Nominatim(user_agent="energy_infrastructure_ai")
        location_data = geolocator.geocode(location)
        
        if location_data:
            result = {
                'latitude': location_data.latitude,
                'longitude': location_data.longitude,
                'location_name': location_data.address,
                'success': True
            }
        else:
            result = {
                'error': f"Could not find coordinates for '{location}'",
                'success': False
            }
        
        return json.dumps(result)
        
    except Exception as e:
        result = {
            'error': f"Geocoding failed: {str(e)}",
            'success': False
        }
        return json.dumps(result)
    

def assess_seismic_risk_at_location(run_context: RunContext[DataSourceTracker], 
                                   latitude: float, longitude: float, 
                                   radius_km: float = 50.0) -> str:
    """
    Assess seismic risk by counting earthquake events within a radius of a specific location.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate  
        radius_km: Search radius in kilometers (default: 50km)
    
    Return:
        JSON string with seismic risk assessment and nearby earthquake data
    """
    
    try:
        # Load seismic data
        df_seismic = load_data_and_process("seismic")
        add_data_source(run_context, ["seismic"])
        
        # Create point geometry for target location
        target_point = Point(longitude, latitude)  # Note: lon, lat for Point
        target_gdf = gpd.GeoDataFrame([{'geometry': target_point}], crs='EPSG:4326')
        
        # Convert to UTM for accurate distance calculation
        utm_crs = target_gdf.estimate_utm_crs()
        target_utm = target_gdf.to_crs(utm_crs)
        seismic_utm = df_seismic.to_crs(utm_crs)
        
        # Create buffer around target point
        buffer_distance = radius_km * 1000  # convert km to meters
        target_buffer = target_utm.copy()
        target_buffer['geometry'] = target_buffer['geometry'].buffer(buffer_distance)
        
        # Find seismic events within buffer
        nearby_seismic = gpd.sjoin(seismic_utm, target_buffer, predicate='within')
        nearby_seismic = nearby_seismic.to_crs('EPSG:4326')
        
        # Calculate risk score
        seismic_count = len(nearby_seismic)
        seismic_risk_score = min(seismic_count / 10.0, 1.0)  # Normalize to max 10 events
        
        # Determine risk level
        if seismic_risk_score < 0.3:
            risk_level = "LOW"
        elif seismic_risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Get details of nearby events
        event_details = []
        for _, event in nearby_seismic.head(10).iterrows():  # Limit to 10 closest events
            event_details.append({
                'name': event.get('Name', 'Unknown'),
                'latitude': event.geometry.y,
                'longitude': event.geometry.x
            })
        
        result = {
            'seismic_count': seismic_count,
            'seismic_risk_score': round(seismic_risk_score, 3),
            'risk_level': risk_level,
            'radius_km': radius_km,
            'assessment_location': {'latitude': latitude, 'longitude': longitude},
            'nearby_events': event_details,
            'assessment_summary': f"Found {seismic_count} seismic events within {radius_km}km. Risk level: {risk_level}"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        result = {
            'error': f"Seismic risk assessment failed: {str(e)}",
            'seismic_count': 0,
            'seismic_risk_score': 0.0,
            'risk_level': "UNKNOWN"
        }
        return json.dumps(result)


def assess_infrastructure_proximity(run_context: RunContext[DataSourceTracker],
                                   latitude: float, longitude: float,
                                   infrastructure_types: str = "wells,pipelines,offshore_fields,windfarms",
                                   radius_km: float = 50.0) -> str:
    """
    Assess infrastructure proximity risk by counting existing infrastructure within a radius.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        infrastructure_types: Comma-separated infrastructure types to check (wells,pipelines,offshore_fields)
        radius_km: Search radius in kilometers (default: 50km)
    
    Return:
        JSON string with infrastructure proximity assessment
    """
    
    try:
        # Parse infrastructure types
        if isinstance(infrastructure_types, str):
            infrastructure_list = [t.strip() for t in infrastructure_types.split(',')]
        else:
            # Fallback option
            infrastructure_list = ["wells", "pipelines", "offshore_fields", "windfarms"]
        
        # Validate infrastructure types
        valid_types = ["wells", "pipelines", "offshore_fields", "licences", "drilling", "windfarms"]
        infrastructure_list = [t for t in infrastructure_list if t in valid_types]
        
        if not infrastructure_list:
            infrastructure_list = ["wells", "pipelines", "offshore_fields", "windfarms"]
        
        add_data_source(run_context, infrastructure_list)
        # for infra_name in infrastructure_list:
        #     add_data_source(run_context, infra_name)
        
        # Create target point
        target_point = Point(longitude, latitude)
        target_gdf = gpd.GeoDataFrame([{'geometry': target_point}], crs='EPSG:4326')
        utm_crs = target_gdf.estimate_utm_crs()
        target_utm = target_gdf.to_crs(utm_crs)
        
        # Create buffer
        buffer_distance = radius_km * 1000
        target_buffer = target_utm.copy()
        target_buffer['geometry'] = target_buffer['geometry'].buffer(buffer_distance)
        
        # Check each infrastructure type
        infrastructure_details = {}
        total_infrastructure_count = 0
        
        for infra_type in infrastructure_list:
            df_infra = load_data_and_process(infra_type)
            infra_utm = df_infra.to_crs(utm_crs)
            
            # Find nearby infrastructure
            # nearby_infra = gpd.sjoin(infra_utm, target_buffer, predicate='within')
            nearby_infra = gpd.sjoin(infra_utm, target_buffer, predicate='intersects')
            count = len(nearby_infra)
            
            infrastructure_details[infra_type] = {
                'count': count,
                'examples': [
                    {
                        'name': row.get('Name', 'Unknown'),
                        'latitude': row.geometry.y if hasattr(row.geometry, 'y') else None,
                        'longitude': row.geometry.x if hasattr(row.geometry, 'x') else None
                    }
                    for _, row in nearby_infra.to_crs('EPSG:4326').head(5).iterrows()
                ]
            }
            
            total_infrastructure_count += count
        
        # Calculate risk score
        infrastructure_risk_score = min(total_infrastructure_count / 20.0, 1.0)  # Normalize to max 20 items
        
        # Determine risk level
        if infrastructure_risk_score < 0.3:
            risk_level = "LOW"
        elif infrastructure_risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        result = {
            'total_infrastructure_count': total_infrastructure_count,
            'infrastructure_risk_score': round(infrastructure_risk_score, 3),
            'risk_level': risk_level,
            'radius_km': radius_km,
            'assessment_location': {'latitude': latitude, 'longitude': longitude},
            'infrastructure_breakdown': infrastructure_details,
            'assessment_summary': f"Found {total_infrastructure_count} infrastructure features within {radius_km}km. Risk level: {risk_level}"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        result = {
            'error': f"Infrastructure proximity assessment failed: {str(e)}",
            'total_infrastructure_count': 0,
            'infrastructure_risk_score': 0.0,
            'risk_level': "UNKNOWN"
        }
        return json.dumps(result)


def calculate_overall_risk_score(run_context: RunContext[DataSourceTracker],
                                seismic_risk: float, infrastructure_risk: float,
                                environmental_risk: float = 0.3) -> str:
    """
    Calculate overall risk score from individual risk components and provide recommendations.
    
    Args:
        seismic_risk: Seismic risk score (0.0-1.0)
        infrastructure_risk: Infrastructure proximity risk score (0.0-1.0)
        environmental_risk: Environmental risk score (0.0-1.0, default: 0.3)
    
    Return:
        JSON string with overall risk assessment and recommendations
    """
    
    try:
        # Validate inputs
        seismic_risk = max(0.0, min(1.0, seismic_risk))
        infrastructure_risk = max(0.0, min(1.0, infrastructure_risk))
        environmental_risk = max(0.0, min(1.0, environmental_risk))
        
        # Weighted overall score
        weights = {'seismic': 0.4, 'infrastructure': 0.3, 'environmental': 0.3}
        overall_risk = (weights['seismic'] * seismic_risk + 
                       weights['infrastructure'] * infrastructure_risk + 
                       weights['environmental'] * environmental_risk)
        
        # Risk categories
        if overall_risk < 0.3:
            risk_level = "LOW"
            risk_color = "green"
        elif overall_risk < 0.6:
            risk_level = "MEDIUM"
            risk_color = "yellow"
        else:
            risk_level = "HIGH"
            risk_color = "red"
        
        # Generate recommendations
        recommendations = []
        if seismic_risk > 0.5:
            recommendations.append("High seismic activity detected - consider enhanced seismic monitoring and earthquake-resistant design")
        if infrastructure_risk > 0.5:
            recommendations.append("High infrastructure density - ensure coordination with existing facilities and assess cumulative impacts")
        if environmental_risk > 0.5:
            recommendations.append("Environmental sensitivity detected - conduct detailed environmental impact assessment")
        if overall_risk < 0.3:
            recommendations.append("Location shows favorable conditions for infrastructure development")
        if overall_risk > 0.7:
            recommendations.append("Consider alternative locations due to high cumulative risk")
        
        # Risk mitigation strategies
        mitigation_strategies = []
        if seismic_risk > 0.3:
            mitigation_strategies.append("Implement real-time seismic monitoring systems")
        if infrastructure_risk > 0.3:
            mitigation_strategies.append("Coordinate with existing infrastructure operators")
        if environmental_risk > 0.3:
            mitigation_strategies.append("Develop comprehensive environmental management plan")
        
        result = {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'component_risks': {
                'seismic': round(seismic_risk, 3),
                'infrastructure': round(infrastructure_risk, 3),
                'environmental': round(environmental_risk, 3)
            },
            'risk_weights': weights,
            'recommendations': recommendations,
            'mitigation_strategies': mitigation_strategies,
            'assessment_summary': f"Overall risk level: {risk_level} (score: {overall_risk:.2f}/1.0)"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        result = {
            'error': f"Risk calculation failed: {str(e)}",
            'overall_risk_score': 0.5,
            'risk_level': "UNKNOWN"
        }
        return json.dumps(result)


def create_risk_assessment_map(run_context: RunContext[DataSourceTracker],
                              latitude: float, longitude: float,
                              location_name: str = "Assessment Location",
                              risk_level: str = "MEDIUM",
                              radius_km: float = 50.0) -> str:
    """
    Create an interactive map visualization for risk assessment results.
    Shows assessment location, radius, and all nearby infrastructure.
    
    Args:
        latitude: Assessment location latitude
        longitude: Assessment location longitude
        location_name: Name/description of the assessment location
        risk_level: Risk level (LOW/MEDIUM/HIGH) for marker color
        radius_km: Assessment radius to display (default: 50km)
    
    Return:
        JSON string with map HTML and summary
    """
    
    try:
        # Create map centered on assessment location
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=8,
            tiles="CartoDB positron"
        )
        
        # Determine marker color based on risk level
        color_map = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        marker_color = color_map.get(risk_level.upper(), 'blue')
        
        # Add assessment location marker
        folium.Marker(
            location=[latitude, longitude],
            popup=f"""
            <b>Risk Assessment Location</b><br>
            Location: {location_name}<br>
            Coordinates: {latitude:.4f}, {longitude:.4f}<br>
            Risk Level: <b style='color:{marker_color}'>{risk_level}</b>
            """,
            tooltip=f"Assessment Location: {risk_level} Risk",
            icon=folium.Icon(color=marker_color, icon='star')
        ).add_to(m)
        
        # Add assessment radius circle
        folium.Circle(
            location=[latitude, longitude],
            radius=radius_km * 1000,  # Convert km to meters
            popup=f"Assessment Radius: {radius_km} km",
            color='blue',
            weight=2,
            fill=False,
            dashArray='5, 5'
        ).add_to(m)
        
        # Create target point for spatial queries
        target_point = Point(longitude, latitude)
        target_gdf = gpd.GeoDataFrame([{'geometry': target_point}], crs='EPSG:4326')
        utm_crs = target_gdf.estimate_utm_crs()
        target_utm = target_gdf.to_crs(utm_crs)
        buffer_distance = radius_km * 1000
        target_buffer = target_utm.copy()
        target_buffer['geometry'] = target_buffer['geometry'].buffer(buffer_distance)
        
        infrastructure_counts = {}
        
        # Add seismic events
        try:
            df_seismic = load_data_and_process("seismic")
            seismic_utm = df_seismic.to_crs(utm_crs)
            nearby_seismic = gpd.sjoin(seismic_utm, target_buffer, predicate='intersects')
            nearby_seismic = nearby_seismic.to_crs('EPSG:4326')
            
            # Add seismic markers (limit to 20 for performance)
            seismic_count = 0
            for idx, row in nearby_seismic.head(20).iterrows():
                if hasattr(row.geometry, 'y'):
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=4,
                        popup=f"Seismic Event: {row.get('Name', 'Unknown')}",
                        color='red',
                        fillColor='red',
                        fillOpacity=0.7,
                        tooltip="Seismic Event"
                    ).add_to(m)
                    seismic_count += 1
            
            infrastructure_counts['seismic'] = len(nearby_seismic)
            
        except Exception as e:
            print(f"Could not add seismic data to map: {e}")
            infrastructure_counts['seismic'] = 0
        
        # Add wells
        try:
            df_wells = load_data_and_process("wells")
            wells_utm = df_wells.to_crs(utm_crs)
            nearby_wells = gpd.sjoin(wells_utm, target_buffer, predicate='intersects')
            nearby_wells = nearby_wells.to_crs('EPSG:4326')
            
            # Add well markers (limit to 30 for performance)
            well_count = 0
            for idx, row in nearby_wells.head(30).iterrows():
                if hasattr(row.geometry, 'y'):
                    status = row.get('ORIGINSTAT', 'Unknown')
                    well_color = 'darkblue' if status != 'Decommissioned' else 'lightblue'
                    
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=f"""
                        <b>Well: {row.get('Name', 'Unknown')}</b><br>
                        Status: {status}<br>
                        Coordinates: {row.geometry.y:.4f}, {row.geometry.x:.4f}
                        """,
                        icon=folium.Icon(color=well_color, icon='tint', prefix='fa'),
                        tooltip=f"Well: {row.get('Name', 'Unknown')}"
                    ).add_to(m)
                    well_count += 1
            
            infrastructure_counts['wells'] = len(nearby_wells)
            
        except Exception as e:
            print(f"Could not add wells data to map: {e}")
            infrastructure_counts['wells'] = 0
        
        # Add pipelines
        try:
            df_pipelines = load_data_and_process("pipelines")
            pipelines_utm = df_pipelines.to_crs(utm_crs)
            nearby_pipelines = gpd.sjoin(pipelines_utm, target_buffer, predicate='intersects')
            nearby_pipelines = nearby_pipelines.to_crs('EPSG:4326')
            
            # Add pipeline lines (limit to 50 for performance)
            pipeline_count = 0
            for idx, row in nearby_pipelines.head(50).iterrows():
                try:
                    geom = row['geometry']
                    name = row.get('Name', f'Pipeline {idx}')
                    
                    if geom.geom_type == 'LineString':
                        coords = list(geom.coords)
                        folium_coords = [[lat, lon] for lon, lat in coords]
                        
                        folium.PolyLine(
                            locations=folium_coords,
                            popup=f"Pipeline: {name}",
                            tooltip=f"Pipeline: {name}",
                            color='cyan',
                            weight=3,
                            opacity=0.8
                        ).add_to(m)
                        pipeline_count += 1
                        
                    elif geom.geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            coords = list(line.coords)
                            folium_coords = [[lat, lon] for lon, lat in coords]
                            
                            folium.PolyLine(
                                locations=folium_coords,
                                popup=f"Pipeline: {name}",
                                tooltip=f"Pipeline: {name}",
                                color='cyan',
                                weight=3,
                                opacity=0.8
                            ).add_to(m)
                        pipeline_count += 1
                        
                except Exception as e:
                    print(f"Error plotting pipeline {idx}: {e}")
                    continue
            
            infrastructure_counts['pipelines'] = len(nearby_pipelines)
            
        except Exception as e:
            print(f"Could not add pipeline data to map: {e}")
            infrastructure_counts['pipelines'] = 0
        
        # Add offshore fields
        try:
            df_fields = load_data_and_process("offshore_fields")
            fields_utm = df_fields.to_crs(utm_crs)
            nearby_fields = gpd.sjoin(fields_utm, target_buffer, predicate='intersects')
            nearby_fields = nearby_fields.to_crs('EPSG:4326')
            
            # Add field polygons
            field_count = 0
            for idx, row in nearby_fields.head(20).iterrows():
                try:
                    geom = row['geometry']
                    name = row.get('Name', f'Field {idx}')
                    
                    if geom.geom_type == 'Polygon':
                        exterior_coords = list(geom.exterior.coords)
                        folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                        
                        folium.Polygon(
                            locations=folium_coords,
                            popup=f"Offshore Field: {name}",
                            tooltip=f"Field: {name}",
                            color='purple',
                            weight=2,
                            opacity=0.8,
                            fillColor='purple',
                            fillOpacity=0.3
                        ).add_to(m)
                        field_count += 1
                        
                    elif geom.geom_type == 'MultiPolygon':
                        for polygon in geom.geoms:
                            exterior_coords = list(polygon.exterior.coords)
                            folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                            
                            folium.Polygon(
                                locations=folium_coords,
                                popup=f"Offshore Field: {name}",
                                tooltip=f"Field: {name}",
                                color='purple',
                                weight=2,
                                opacity=0.8,
                                fillColor='purple',
                                fillOpacity=0.3
                            ).add_to(m)
                        field_count += 1
                        
                except Exception as e:
                    print(f"Error plotting field {idx}: {e}")
                    continue
            
            infrastructure_counts['offshore_fields'] = len(nearby_fields)
            
        except Exception as e:
            print(f"Could not add offshore fields data to map: {e}")
            infrastructure_counts['offshore_fields'] = 0
        
        # Add windfarms - single color treatment
        try:
            df_windfarms = load_data_and_process("windfarms")
            windfarms_utm = df_windfarms.to_crs(utm_crs)
            nearby_windfarms = gpd.sjoin(windfarms_utm, target_buffer, predicate='intersects')
            nearby_windfarms = nearby_windfarms.to_crs('EPSG:4326')
            
            # Add windfarm polygons - all same style
            windfarm_count = 0
            for idx, row in nearby_windfarms.head(20).iterrows():
                try:
                    geom = row['geometry']
                    name = row.get('Name', f'Windfarm {idx}')
                    
                    if geom.geom_type == 'Polygon':
                        exterior_coords = list(geom.exterior.coords)
                        folium_coords = [[lat, lon] for lon, lat in exterior_coords]
                        
                        folium.Polygon(
                            locations=folium_coords,
                            popup=f"Active Windfarm: {name}",
                            tooltip=f"Windfarm: {name}",
                            color='darkgreen',
                            weight=2,
                            opacity=0.8,
                            fillColor='lightgreen',
                            fillOpacity=0.4
                        ).add_to(m)
                        windfarm_count += 1
                        
                except Exception as e:
                    print(f"Error plotting windfarm {idx}: {e}")
                    continue
            
            infrastructure_counts['windfarms'] = len(nearby_windfarms)
            
        except Exception as e:
            print(f"Could not add windfarm data to map: {e}")
            infrastructure_counts['windfarms'] = 0


        # Enhanced legend with actual counts
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 280px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Risk Assessment Map</h4>
        <p><i class="fa fa-star" style="color:{marker_color}"></i> Assessment Location ({risk_level} Risk)</p>
        <p><i class="fa fa-circle" style="color:red"></i> Seismic Events ({infrastructure_counts.get('seismic', 0)})</p>
        <p><i class="fa fa-tint" style="color:darkblue"></i> Wells ({infrastructure_counts.get('wells', 0)})</p>
        <p><span style="color:cyan; font-weight:bold;">━━</span> Pipelines ({infrastructure_counts.get('pipelines', 0)})</p>
        <p><span style="color:purple;">▬</span> Offshore Fields ({infrastructure_counts.get('offshore_fields', 0)})</p>
        <p><span style="color:darkgreen;">▬</span> Windfarms ({infrastructure_counts.get('windfarms', 0)})</p>
        <p><span style="color:blue; font-weight:bold;">- - -</span> Assessment Radius ({radius_km} km)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        map_html = m.get_root().render()
        
        # Create summary with actual infrastructure counts
        total_infrastructure = sum(infrastructure_counts.values())
        summary = f"""
**RISK ASSESSMENT MAP GENERATED**

**Location:** {location_name}  
**Coordinates:** {latitude:.4f}, {longitude:.4f}  
**Risk Level:** {risk_level}  
**Assessment Radius:** {radius_km} km  

**Infrastructure Detected:**
- Seismic Events: {infrastructure_counts.get('seismic', 0)}
- Wells: {infrastructure_counts.get('wells', 0)}
- Pipelines: {infrastructure_counts.get('pipelines', 0)}
- Offshore Fields: {infrastructure_counts.get('offshore_fields', 0)}
- Windfarms: {infrastructure_counts.get('windfarms', 0)}
- **Total: {total_infrastructure} features**

The interactive map displays all detected infrastructure within the assessment radius with different colors and symbols for each type. Click on markers and lines for detailed information.
        """
        
        result = {
            'report': summary,
            'map_html': map_html
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_summary = f"""
**MAP GENERATION FAILED**

Location: {location_name}
Error: {str(e)}

Unable to create comprehensive risk assessment map visualization.
        """
        
        result = {
            'report': error_summary,
            'map_html': '<div style="height: 400px; display: flex; align-items: center; justify-content: center; color: red;">Map creation failed</div>'
        }
        
        return json.dumps(result)
    

# === Tools relating to exploration planner ===
def create_exploration_map(top_sites: gpd.GeoDataFrame, 
                          all_grid: gpd.GeoDataFrame, 
                          weights: Dict[str, float]) -> str:
    """Create map using strategic spacing to show all data without clutter."""
    
    try:
        print(f"DEBUG: Creating spaced visualization with {len(top_sites)} top sites and {len(all_grid)} grid cells")
        
        # UK-centered coordinates
        uk_center_lat = 55.5
        uk_center_lon = -2.0
        
        # Create map
        m = folium.Map(
            location=[uk_center_lat, uk_center_lon],
            zoom_start=6,  # Slightly more zoomed for detail
            tiles="OpenStreetMap"
        )
        
        # Add terrain layer
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Terrain",
            name="Terrain",
            overlay=False,
            control=True
        ).add_to(m)
        
        # STRATEGIC APPROACH: Show representative samples from each suitability category
        max_score = all_grid['suitability_score'].max()
        min_score = all_grid['suitability_score'].min()
        
        # Categorize all grid cells
        def categorize_suitability(score, max_score, min_score):
            if max_score == min_score:
                return "moderate"
            normalized = (score - min_score) / (max_score - min_score)
            if normalized <= 0.2:
                return "excellent"
            elif normalized <= 0.4:
                return "very_good" 
            elif normalized <= 0.6:
                return "moderate"
            elif normalized <= 0.8:
                return "poor"
            else:
                return "very_poor"
        
        all_grid['category'] = all_grid['suitability_score'].apply(
            lambda x: categorize_suitability(x, max_score, min_score)
        )
        
        # Sample from each category to maintain representation while reducing clutter
        category_samples = {}
        sample_sizes = {
            'excellent': min(500, len(all_grid[all_grid['category'] == 'excellent'])),
            'very_good': min(400, len(all_grid[all_grid['category'] == 'very_good'])),
            'moderate': min(800, len(all_grid[all_grid['category'] == 'moderate'])),  # Show more mediocre
            'poor': min(300, len(all_grid[all_grid['category'] == 'poor'])),
            'very_poor': min(200, len(all_grid[all_grid['category'] == 'very_poor']))
        }
        
        for category, sample_size in sample_sizes.items():
            category_data = all_grid[all_grid['category'] == category]
            if len(category_data) > 0:
                if len(category_data) > sample_size:
                    # Stratified sampling across the geographic area
                    category_samples[category] = category_data.sample(n=sample_size, random_state=42)
                else:
                    category_samples[category] = category_data
        
        # Define visual properties for each category
        category_styles = {
            'excellent': {'color': '#006400', 'size': 6, 'opacity': 0.8},
            'very_good': {'color': '#32CD32', 'size': 5, 'opacity': 0.7},
            'moderate': {'color': '#FFD700', 'size': 4, 'opacity': 0.6},  # Golden yellow for mediocre
            'poor': {'color': '#FF8C00', 'size': 3, 'opacity': 0.6},
            'very_poor': {'color': '#FF0000', 'size': 3, 'opacity': 0.7}
        }
        
        # Add markers for each category
        total_markers = 0
        for category, data in category_samples.items():
            style = category_styles[category]
            
            for idx, cell in data.iterrows():
                try:
                    lat = cell.get('center_lat', 0)
                    lon = cell.get('center_lon', 0)
                    score = cell.get('suitability_score', 0)
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        popup=f"Suitability: {category.replace('_', ' ').title()}<br>Score: {score:.3f}<br>Cell: {cell.get('cell_id', idx)}",
                        radius=style['size'],
                        color='black',
                        weight=0.5,
                        fill=True,
                        fill_opacity=style['opacity'],
                        fill_color=style['color'],
                        tooltip=f"{category.replace('_', ' ').title()}"
                    ).add_to(m)
                    
                    total_markers += 1
                    
                except Exception as e:
                    continue
        
        print(f"DEBUG: Added {total_markers} representative markers")
        
        # Add top candidates with prominent styling
        for idx, site in top_sites.iterrows():
            try:
                lat = site.get('center_lat', 0)
                lon = site.get('center_lon', 0)
                rank = site.get('rank', idx + 1)
                score = site.get('suitability_score', 0)
                
                popup_html = f"""
                <div style='width: 220px;'>
                <h4 style='color: darkgreen; text-align: center;'>⭐ TOP EXPLORATION SITE</h4>
                <b>Rank:</b> #{int(rank)}<br>
                <b>Suitability Score:</b> {score:.3f}<br>
                <b>Coordinates:</b> {lat:.3f}°N, {abs(lon):.3f}°W<br>
                <hr>
                <b>Risk Assessment:</b><br>
                • Seismic: {site.get('seismic_score', 0):.3f}<br>
                • Ecological: {site.get('ecological_score', 0):.3f}<br>
                • Infrastructure: {site.get('infrastructure_score', 0):.3f}
                </div>
                """
                
                # Large prominent marker
                folium.CircleMarker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    radius=12,
                    color='gold',
                    weight=3,
                    fill=True,
                    fill_opacity=0.9,
                    fill_color='darkgreen',
                    tooltip=f"⭐ TOP SITE #{int(rank)}"
                ).add_to(m)
                
            except Exception as e:
                continue
        
        # Add UK cities
        uk_cities = [
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Edinburgh", "lat": 55.9533, "lon": -3.1883},
            {"name": "Aberdeen", "lat": 57.1497, "lon": -2.0943},
            {"name": "Newcastle", "lat": 54.9783, "lon": -1.6178}
        ]
        
        for city in uk_cities:
            folium.CircleMarker(
                location=[city["lat"], city["lon"]],
                radius=4,
                popup=f"🏙️ {city['name']}",
                color='black',
                weight=2,
                fill=True,
                fill_opacity=0.9,
                fill_color='white',
                tooltip=city["name"]
            ).add_to(m)
        
        # Minimal existing wells
        try:
            df_wells = load_data_and_process("wells")
            wells_sample = df_wells.sample(n=min(15, len(df_wells)), random_state=42)
            
            for idx, well in wells_sample.iterrows():
                if hasattr(well.geometry, 'y'):
                    folium.CircleMarker(
                        location=[well.geometry.y, well.geometry.x],
                        radius=2,
                        popup=f"Existing Well: {well.get('Name', 'Unknown')}",
                        color='navy',
                        weight=1,
                        fill=True,
                        fill_opacity=0.7,
                        fill_color='lightblue',
                        tooltip="Infrastructure"
                    ).add_to(m)
        except:
            pass
        
        # Enhanced legend showing actual representation
        category_counts = {cat: len(data) for cat, data in category_samples.items()}
        weights_display = ", ".join([f"{k}: {v:.0%}" for k, v in weights.items()])
        
        legend_html = f'''
        <div style="position: fixed; top: 5px; right: 10px; width: 200px; height: 180px; 
                    background-color: rgba(255,255,255,0.95); border:1px solid #333; z-index:9999; 
                    font-size:10px; padding: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    border-radius: 4px; font-family: Arial, sans-serif;">
        <h4 style="margin:0 0 6px 0; color:#2E8B57; text-align:center; font-size:12px;">UK Exploration Planner</h4>

        <div style="margin:4px 0;">
        <div style="display:flex; align-items:center; margin:2px 0;">
            <span style="width:8px; height:8px; background:#006400; border:1px solid black; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Excellent ({category_counts.get('excellent', 0)})</span>
        </div>
        <div style="display:flex; align-items:center; margin:2px 0;">
            <span style="width:7px; height:7px; background:#32CD32; border:1px solid black; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Very Good ({category_counts.get('very_good', 0)})</span>
        </div>
        <div style="display:flex; align-items:center; margin:2px 0;">
            <span style="width:6px; height:6px; background:#FFD700; border:1px solid black; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Moderate ({category_counts.get('moderate', 0)})</span>
        </div>
        <div style="display:flex; align-items:center; margin:2px 0;">
            <span style="width:5px; height:5px; background:#FF8C00; border:1px solid black; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Poor ({category_counts.get('poor', 0)})</span>
        </div>
        <div style="display:flex; align-items:center; margin:2px 0;">
            <span style="width:5px; height:5px; background:#FF0000; border:1px solid black; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Very Poor ({category_counts.get('very_poor', 0)})</span>
        </div>
        </div>

        <div style="margin:6px 0 4px 0;">
        <div style="display:flex; align-items:center; margin:1px 0;">
            <span style="width:10px; height:10px; background:darkgreen; border:2px solid gold; border-radius:50%; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Top Sites ({len(top_sites)})</span>
        </div>
        <div style="display:flex; align-items:center; margin:1px 0;">
            <span style="width:5px; height:5px; background:white; border:1px solid black; border-radius:50%; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">UK Cities</span>
        </div>
        <div style="display:flex; align-items:center; margin:1px 0;">
            <span style="width:4px; height:4px; background:lightblue; border:1px solid navy; border-radius:50%; margin-right:4px; display:inline-block;"></span>
            <span style="font-size:9px;">Wells</span>
        </div>
        </div>

        <div style="margin:4px 0 0 0; font-size:8px; color:#666; text-align:center;">
        {total_markers} points | {weights_display}
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        
        map_html = m.get_root().render()
        map_size_mb = len(map_html) / (1024 * 1024)
        print(f"DEBUG: Generated strategic sampling map HTML size: {map_size_mb:.2f} MB")
        
        return map_html
        
    except Exception as e:
        print(f"ERROR in strategic sampling map creation: {e}")
        import traceback
        traceback.print_exc()
        return f'<div style="height: 400px; color: red; text-align: center; padding-top: 150px;">Strategic sampling map creation failed: {str(e)}</div>'


def plan_low_impact_exploration_sites(run_context: RunContext[DataSourceTracker],
                                      goal: str = "minimize environmental impact",
                                      scenario_name: str = "environment_focus",
                                      adjust_safety: float = 0.0,
                                      adjust_technical: float = 0.0, 
                                      adjust_economic: float = 0.0,
                                      adjust_environment: float = 0.0,
                                      max_seismic_risk: float = 0.3,
                                      max_ecological_sensitivity: float = 0.4,
                                      min_infrastructure_proximity: float = 0.1,
                                      num_sites: int = 10,
                                      cell_size_km: float = 5.0) -> str:
    """
    Autonomous exploration planning that identifies low-environmental-impact candidate locations
    for new exploration/drilling, balancing seismic risk, ecological sensitivity, and infrastructure proximity.
    
    Available scenarios from config.py:
    - "balanced": All objectives weighted equally at 0.25
    - "economic_focus": Economic weighted at 0.5, others at 0.1-0.2  
    - "safety_focus": Safety weighted at 0.5, others at 0.1-0.2
    - "technical_focus": Technical weighted at 0.5, others at 0.1-0.2
    - "environment_focus": Environment weighted at 0.5, others at 0.1-0.2
    
    Args:
        goal: Planning objective description (used for report generation)
        scenario_name: Base scenario from SCENARIOS in config.py
        adjust_safety: Adjustment to safety weight (e.g., +0.1 to increase by 0.1)
        adjust_technical: Adjustment to technical weight  
        adjust_economic: Adjustment to economic weight
        adjust_environment: Adjustment to environment weight
        max_seismic_risk: Maximum acceptable seismic risk score (0.0-1.0)
        max_ecological_sensitivity: Maximum acceptable ecological sensitivity (0.0-1.0)
        min_infrastructure_proximity: Minimum required infrastructure proximity (0.0-1.0)
        num_sites: Number of candidate sites to return
        cell_size_km: Grid cell size in kilometers
    
    Examples:
        - "Use safety focus with more emphasis on technical aspects" → scenario_name="safety_focus", adjust_technical=0.2
        - "Balanced approach but reduce economic importance" → scenario_name="balanced", adjust_economic=-0.1
    
    Return:
        JSON string with exploration plan, map, and detailed site analyses
    """
    
    print(f"🎯 Planning exploration sites with goal: {goal}")
    print(f"📋 Base scenario: {scenario_name}")
    
    try:
        # Import the grid system
        from grid_system import ExplorationGridSystem
        from config import SCENARIOS
        
        # Add data sources to tracker
        add_data_source(run_context, ["seismic", "wells", "pipelines", "offshore_fields"])
        
        # Get base weights from config scenarios
        if scenario_name not in SCENARIOS:
            available_scenarios = ", ".join(SCENARIOS.keys())
            return json.dumps({
                'error': f'Unknown scenario "{scenario_name}". Available: {available_scenarios}',
                'report': f'Scenario "{scenario_name}" not found in configuration.'
            })
        
        # Start with base scenario weights
        weights = SCENARIOS[scenario_name].copy()
        print(f"📊 Base weights from {scenario_name}: {weights}")
        
        # Apply user adjustments
        adjustments = {}
        if adjust_safety != 0.0:
            weights['safety'] = max(0, weights['safety'] + adjust_safety)
            adjustments['safety'] = adjust_safety
        if adjust_technical != 0.0:
            weights['technical'] = max(0, weights['technical'] + adjust_technical)
            adjustments['technical'] = adjust_technical
        if adjust_economic != 0.0:
            weights['economic'] = max(0, weights['economic'] + adjust_economic)
            adjustments['economic'] = adjust_economic
        if adjust_environment != 0.0:
            weights['environment'] = max(0, weights['environment'] + adjust_environment)
            adjustments['environment'] = adjust_environment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight == 0:
            return json.dumps({
                'error': 'All weights are zero after adjustments',
                'report': 'Weight adjustments resulted in zero total weight.'
            })
        
        weights = {k: v / total_weight for k, v in weights.items()}
        
        if adjustments:
            print(f"🔧 Applied adjustments: {adjustments}")
        print(f"⚖️ Final normalized weights: {weights}")
        
        # Map weights to grid scoring system (safety→seismic, environment→ecological, etc.)
        mcda_weights = {
            'seismic': weights['safety'],
            'ecological': weights['environment'], 
            'infrastructure': weights['technical'] + weights['economic']  # Combine technical and economic for infrastructure
        }
        
        # Renormalize MCDA weights
        mcda_total = sum(mcda_weights.values())
        mcda_weights = {k: v / mcda_total for k, v in mcda_weights.items()}
        
        print(f"🔄 Mapped to MCDA weights: {mcda_weights}")
        print(f"📐 Constraints: seismic≤{max_seismic_risk}, ecological≤{max_ecological_sensitivity}, infrastructure≥{min_infrastructure_proximity}")
        
        # Initialize grid system
        grid_system = ExplorationGridSystem(cell_size_km=cell_size_km)
        
        # Calculate all scores
        print("🔢 Calculating grid scores...")
        grid_with_scores = grid_system.get_scored_grid()
        
        # Apply constraints to filter suitable cells
        suitable_cells = grid_with_scores[
            (grid_with_scores['seismic_score'] <= max_seismic_risk) &
            (grid_with_scores['ecological_score'] <= max_ecological_sensitivity) &
            (grid_with_scores['infrastructure_score'] >= min_infrastructure_proximity)
        ].copy()
        
        if len(suitable_cells) == 0:
            return json.dumps({
                'error': 'No sites meet the specified constraints',
                'recommendation': 'Try relaxing constraints or expanding search area',
                'report': 'No suitable exploration sites found with current criteria.',
                'scenario_used': scenario_name,
                'weights_applied': weights,
                'adjustments_made': adjustments
            })
        
        print(f"🔍 Found {len(suitable_cells)} cells meeting constraints")
        
        # Run MCDA on suitable cells using the calculated weights
        print("🎯 Running MCDA analysis...")
        ranked_cells = grid_system.run_mcda_analysis(mcda_weights)
        
        # Filter to only suitable cells and get top candidates
        ranked_suitable = ranked_cells[
            (ranked_cells['seismic_score'] <= max_seismic_risk) &
            (ranked_cells['ecological_score'] <= max_ecological_sensitivity) &
            (ranked_cells['infrastructure_score'] >= min_infrastructure_proximity)
        ].head(num_sites)
        
        if len(ranked_suitable) == 0:
            return json.dumps({
                'error': 'No suitable sites after MCDA ranking',
                'report': 'MCDA analysis found no sites meeting criteria.',
                'scenario_used': scenario_name,
                'weights_applied': weights
            })
        
        # Create interactive map
        print("🗺️ Creating interactive map...")
        map_html = create_exploration_map(ranked_suitable, grid_with_scores, mcda_weights)
        
        # Generate scenario description
        scenario_description = f"{scenario_name}"
        if adjustments:
            adj_desc = ", ".join([f"{k} {'+' if v > 0 else ''}{v}" for k, v in adjustments.items()])
            scenario_description += f" (adjusted: {adj_desc})"
        
        # Generate LLM explanations for top sites
        print("📝 Generating site explanations...")
        site_explanations = generate_site_explanations(ranked_suitable.head(5), mcda_weights, scenario_description)
        
        # Create comprehensive report with scenario info
        print("📄 Creating comprehensive report...")
        report = generate_exploration_report_with_scenario(
            ranked_suitable, weights, mcda_weights, scenario_name, adjustments,
            max_seismic_risk, max_ecological_sensitivity, min_infrastructure_proximity
        )
        
        result = {
            'report': report,
            'map_html': map_html,
            'scenario_used': scenario_name,
            'scenario_description': scenario_description,
            'original_weights': SCENARIOS[scenario_name],
            'adjustments_made': adjustments,
            'final_weights': weights,
            'mcda_weights_applied': mcda_weights,
            'total_suitable_sites': len(suitable_cells),
            'top_candidates': len(ranked_suitable),
            'site_explanations': site_explanations,
            'constraints_applied': {
                'max_seismic_risk': max_seismic_risk,
                'max_ecological_sensitivity': max_ecological_sensitivity,
                'min_infrastructure_proximity': min_infrastructure_proximity
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        print(f"❌ Error in exploration planning: {e}")
        import traceback
        traceback.print_exc()
        error_result = {
            'error': f'Exploration planning failed: {str(e)}',
            'report': f'Unable to complete exploration planning due to: {str(e)}',
            'scenario_used': scenario_name if 'scenario_name' in locals() else 'unknown'
        }
        return json.dumps(error_result)


def generate_exploration_report_with_scenario(ranked_sites: gpd.GeoDataFrame, 
                                            original_weights: Dict[str, float],
                                            mcda_weights: Dict[str, float],
                                            scenario_name: str,
                                            adjustments: Dict[str, float],
                                            max_seismic: float,
                                            max_ecological: float, 
                                            min_infrastructure: float) -> str:
    """Generate comprehensive exploration planning report with scenario details."""
    
    report_lines = []
    
    # Header with scenario info
    report_lines.append("# LOW-IMPACT EXPLORATION PLANNING REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**Base Scenario:** {scenario_name}")
    
    if adjustments:
        adj_text = ", ".join([f"{k} {'+' if v >= 0 else ''}{v:.2f}" for k, v in adjustments.items()])
        report_lines.append(f"**Weight Adjustments:** {adj_text}")
    else:
        report_lines.append("**Weight Adjustments:** None")
    
    report_lines.append(f"**Planning Objective:** Minimize environmental impact while maintaining operational feasibility")
    report_lines.append("")
    
    # Scenario weights breakdown
    report_lines.append("## SCENARIO WEIGHT ANALYSIS")
    report_lines.append(f"**Base Scenario ({scenario_name}):**")
    for criterion, weight in original_weights.items():
        report_lines.append(f"- {criterion.title()}: {weight:.1%}")
    
    if adjustments:
        report_lines.append("")
        report_lines.append("**After Adjustments:**")
        final_weights = original_weights.copy()
        for k, v in adjustments.items():
            final_weights[k] = max(0, final_weights[k] + v)
        
        # Show normalized final weights
        total = sum(final_weights.values())
        for criterion, weight in final_weights.items():
            normalized = weight / total if total > 0 else 0
            change = ""
            if criterion in adjustments:
                if adjustments[criterion] > 0:
                    change = " ⬆️"
                elif adjustments[criterion] < 0:
                    change = " ⬇️"
            report_lines.append(f"- {criterion.title()}: {normalized:.1%}{change}")
    
    report_lines.append("")
    report_lines.append("**MCDA Mapping:**")
    for criterion, weight in mcda_weights.items():
        report_lines.append(f"- {criterion.title()}: {weight:.1%}")
    report_lines.append("")
    
    # Constraints section
    report_lines.append("## CONSTRAINTS APPLIED")
    report_lines.append(f"- Maximum Seismic Risk Score: {max_seismic} (lower is safer)")
    report_lines.append(f"- Maximum Ecological Sensitivity Score: {max_ecological} (lower is less environmentally sensitive)")
    report_lines.append(f"- Minimum Infrastructure Proximity Score: {min_infrastructure} (higher means closer to existing infrastructure)")
    report_lines.append("")
    
    # Continue with the rest of the original report structure...
    report_lines.append("## SCORING METHODOLOGY")
    report_lines.append("**Suitability Score Calculation:**")
    report_lines.append("- **Lower suitability scores = BETTER exploration sites**")
    report_lines.append("- Seismic score: 0.0 (no earthquakes) to 1.0 (many earthquakes)")
    report_lines.append("- Ecological score: 0.0 (low sensitivity) to 1.0 (high sensitivity)")
    report_lines.append("- Infrastructure score: 0.0 (isolated) to 1.0 (well-connected)")
    report_lines.append("- Combined using weighted average based on selected scenario")
    report_lines.append("")
    
    # Results summary
    report_lines.append("## RESULTS SUMMARY")
    report_lines.append(f"- **Total Candidate Sites:** {len(ranked_sites)}")
    best_score = ranked_sites['suitability_score'].min()
    worst_score = ranked_sites['suitability_score'].max()
    report_lines.append(f"- **Best Suitability Score:** {best_score:.3f} (lower is better)")
    report_lines.append(f"- **Score Range:** {best_score:.3f} - {worst_score:.3f}")
    report_lines.append("")
    
    # Top candidates table (keep existing table format)
    report_lines.append("## TOP CANDIDATE LOCATIONS (Best to Worst)")
    report_lines.append("| Rank | Cell ID | Coordinates | Suitability↓ | Seismic | Ecological | Infrastructure |")
    report_lines.append("|------|---------|-------------|--------------|---------|------------|----------------|")
    
    for _, site in ranked_sites.head(10).iterrows():
        coords = f"{site['center_lat']:.2f}°N, {abs(site['center_lon']):.2f}°W"
        
        if site['suitability_score'] <= 0.3:
            quality = "⭐ EXCELLENT"
        elif site['suitability_score'] <= 0.5:
            quality = "✅ GOOD"
        elif site['suitability_score'] <= 0.7:
            quality = "⚠️ FAIR"
        else:
            quality = "❌ POOR"
            
        report_lines.append(
            f"| {int(site['rank'])} {quality} | {int(site['cell_id'])} | {coords} | "
            f"{site['suitability_score']:.3f} | {site['seismic_score']:.3f} | "
            f"{site['ecological_score']:.3f} | {site['infrastructure_score']:.3f} |"
        )
    
    report_lines.append("")
    
    # Scenario-specific recommendations
    report_lines.append("## SCENARIO-SPECIFIC INSIGHTS")
    
    if scenario_name == "environment_focus":
        report_lines.append("🌱 **Environment-Focused Analysis:**")
        report_lines.append("   - Sites prioritize minimal ecological impact")
        report_lines.append("   - Economic considerations are secondary")
    elif scenario_name == "economic_focus":
        report_lines.append("💰 **Economic-Focused Analysis:**")
        report_lines.append("   - Sites prioritize proximity to existing infrastructure")
        report_lines.append("   - Environmental impact balanced with cost considerations")
    elif scenario_name == "safety_focus":
        report_lines.append("🛡️ **Safety-Focused Analysis:**")
        report_lines.append("   - Sites prioritize low seismic risk areas")
        report_lines.append("   - Maximum emphasis on operational safety")
    elif scenario_name == "balanced":
        report_lines.append("⚖️ **Balanced Analysis:**")
        report_lines.append("   - Sites represent optimal compromise across all factors")
        report_lines.append("   - No single criterion dominates the selection")
    
    if adjustments:
        report_lines.append("")
        report_lines.append("**Weight Adjustment Impact:**")
        for criterion, adjustment in adjustments.items():
            if adjustment > 0:
                report_lines.append(f"   - Increased emphasis on {criterion} ({adjustment:+.2f})")
            elif adjustment < 0:
                report_lines.append(f"   - Reduced emphasis on {criterion} ({adjustment:+.2f})")
    
    report_lines.append("")
    
    # Strategic recommendations (keep existing)
    avg_ecological = ranked_sites.head(10)['ecological_score'].mean()
    avg_seismic = ranked_sites.head(10)['seismic_score'].mean()
    
    report_lines.append("## ENVIRONMENTAL IMPACT ASSESSMENT")
    report_lines.append(f"- **Average Ecological Sensitivity (Top 10):** {avg_ecological:.3f}")
    report_lines.append(f"- **Average Seismic Risk (Top 10):** {avg_seismic:.3f}")
    
    if avg_ecological < 0.3 and avg_seismic < 0.3:
        impact_level = "LOW IMPACT (Excellent for sustainable exploration)"
    elif avg_ecological < 0.5 and avg_seismic < 0.5:
        impact_level = "MEDIUM IMPACT (Acceptable with mitigation)"
    else:
        impact_level = "HIGH IMPACT (Requires careful consideration)"
    
    report_lines.append(f"- **Overall Environmental Impact Level:** {impact_level}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## STRATEGIC RECOMMENDATIONS")
    
    if best_score < 0.3:
        report_lines.append("🎯 **PROCEED WITH CONFIDENCE:** Excellent low-impact sites identified")
        report_lines.append("   - Sites show low environmental risk and good operational feasibility")
        report_lines.append("   - Standard environmental management protocols should suffice")
    elif best_score < 0.5:
        report_lines.append("⚠️ **PROCEED WITH CAUTION:** Good sites available but require enhanced monitoring")
        report_lines.append("   - Implement enhanced environmental monitoring")
        report_lines.append("   - Consider additional mitigation measures")
    else:
        report_lines.append("🔍 **DETAILED ASSESSMENT REQUIRED:** Sites available but with elevated risks")
        report_lines.append("   - Conduct comprehensive environmental impact assessments")
        report_lines.append("   - Develop robust mitigation strategies")
    
    report_lines.append("")
    report_lines.append("### Immediate Next Steps:")
    report_lines.append("1. Conduct detailed geological surveys for top 3 candidates")
    report_lines.append("2. Initiate comprehensive environmental impact assessments")
    report_lines.append("3. Begin early stakeholder engagement process")
    report_lines.append("4. Develop site-specific environmental management plans")
    
    if adjustments:
        report_lines.append("5. Validate weight adjustments with domain experts")
    
    report_lines.append("")
    
    # Footer
    report_lines.append("---")
    report_lines.append("*Report generated by Low-Impact Exploration Planner*")
    report_lines.append(f"*Using {scenario_name} scenario with MCDA weights: {mcda_weights}*")
    report_lines.append("")
    report_lines.append("**Weight Mapping Explanation:**")
    report_lines.append(f"*Base scenario: {scenario_name} → MCDA implementation:*")
    report_lines.append(f"- Safety ({original_weights['safety']:.1%}) → Seismic Risk ({mcda_weights['seismic']:.1%})")
    report_lines.append(f"- Environment ({original_weights['environment']:.1%}) → Ecological Sensitivity ({mcda_weights['ecological']:.1%})")
    report_lines.append(f"- Technical ({original_weights['technical']:.1%}) + Economic ({original_weights['economic']:.1%}) → Infrastructure Proximity ({mcda_weights['infrastructure']:.1%})")
    report_lines.append("")
    report_lines.append("*Technical and Economic criteria are combined into Infrastructure Proximity*")
    report_lines.append("*because both relate to operational feasibility and cost-effectiveness.*")
    report_lines.append("")
    report_lines.append("*This analysis provides initial screening - detailed site surveys are essential*")
    
    return "\n".join(report_lines)


def generate_site_explanations(top_sites: gpd.GeoDataFrame, 
                              weights: Dict[str, float], 
                              scenario: str) -> List[Dict]:
    """Generate LLM explanations for top exploration sites."""
    
    explanations = []
    
    for idx, site in top_sites.iterrows():
        # Determine risk levels
        seismic_level = "LOW" if site['seismic_score'] < 0.3 else "MEDIUM" if site['seismic_score'] < 0.6 else "HIGH"
        ecological_level = "LOW" if site['ecological_score'] < 0.3 else "MEDIUM" if site['ecological_score'] < 0.6 else "HIGH"
        infrastructure_level = "LOW" if site['infrastructure_score'] < 0.3 else "MEDIUM" if site['infrastructure_score'] < 0.6 else "HIGH"
        
        # Generate reasoning
        reasoning = f"""
Site #{site['rank']} at {site['center_lat']:.3f}°N, {abs(site['center_lon']):.3f}°W shows {seismic_level} seismic risk, 
{ecological_level} ecological sensitivity, and {infrastructure_level} infrastructure proximity. 
This combination gives it a suitability score of {site['suitability_score']:.3f} under the {scenario} scenario.
        """.strip()
        
        # Generate recommendations
        recommendations = []
        if seismic_level != "LOW":
            recommendations.append("Implement enhanced seismic monitoring")
        if ecological_level != "LOW":
            recommendations.append("Conduct detailed environmental impact assessment")
        if infrastructure_level == "LOW":
            recommendations.append("Plan for extended infrastructure development")
        else:
            recommendations.append("Coordinate with existing infrastructure operators")
        
        if not recommendations:
            recommendations.append("Proceed with standard exploration protocols")
        
        explanations.append({
            'site_rank': int(site['rank']),
            'cell_id': int(site['cell_id']),
            'coordinates': f"{site['center_lat']:.3f}°N, {abs(site['center_lon']):.3f}°W",
            'suitability_score': round(site['suitability_score'], 3),
            'risk_assessment': {
                'seismic': seismic_level,
                'ecological': ecological_level,
                'infrastructure': infrastructure_level
            },
            'reasoning': reasoning,
            'recommendations': recommendations
        })
    
    return explanations


# === Tools relating to wind farm exploration planner ===
def plan_global_wind_farm_sites(run_context: RunContext[DataSourceTracker],
                                region: str = "africa",
                                # Explicit bounds parameters (if provided, override region)
                                min_lat: float = None,
                                max_lat: float = None, 
                                min_lon: float = None,
                                max_lon: float = None,
                                location_name: str = None,
                                goal: str = "balance environmental and economic factors",
                                wind_resource_weight: float = 0.4,
                                environmental_weight: float = 0.25,
                                economic_weight: float = 0.25,
                                operational_weight: float = 0.1,
                                min_wind_resource: float = 0.01,  # Lowered from 0.3
                                max_wave_height: float = 12.0,    # Increased from 4.0
                                num_sites: int = 40,
                                cell_size_km: float = 20.0,
                                fast_mode: bool = True,
                                adaptive_constraints: bool = True) -> str:
    """
    Plan offshore wind farm sites for any location.
    
    Args:
        region: Target region (used if explicit bounds not provided)
        min_lat, max_lat, min_lon, max_lon: Explicit geographical bounds
        location_name: Name of the analyzed location
        goal: Planning objective
        wind_resource_weight to operational_weight: Criterion weights (0-1)
        min_wind_resource: Minimum wind resource threshold (relaxed if needed)
        max_wave_height: Maximum wave height threshold (relaxed if needed)
        num_sites: Number of sites to return
        cell_size_km: Grid cell size
        fast_mode: Use optimized algorithms
        adaptive_constraints: Automatically relax constraints if no sites found
    
    Return:
        JSON with results, constraint adjustments, and recommendations
    """
    
    start_time = datetime.now()
    print(f"Starting adaptive wind farm planning for {region}...")
    
    try:
        # Use explicit bounds if all are provided
        bounds_to_use = None
        effective_region = region
        
        if all(param is not None for param in [min_lat, max_lat, min_lon, max_lon]):
            bounds_to_use = {
                'min_lat': min_lat,
                'max_lat': max_lat, 
                'min_lon': min_lon,
                'max_lon': max_lon
            }
            effective_region = location_name or f"Custom Location ({min_lat:.1f},{min_lon:.1f})"
            print(f"Using explicit bounds for: {effective_region}")


        # Initialize planner
        max_cells = 3000 if fast_mode else 8000
        planner = GlobalWindFarmPlanner(
            region=region,
            cell_size_km=cell_size_km,
            max_grid_cells=max_cells,
            custom_bounds=bounds_to_use,
        )
        
        # Create grid
        planner.create_grid()
        
        if len(planner.grid_gdf) == 0:
            return json.dumps({
                'error': 'No offshore grid cells created',
                'region': region,
                'suggestion': 'Try different region or check regional boundaries'
            })
        
        print(f"Grid created: {len(planner.grid_gdf)} cells")
        
        # Calculate scores
        if fast_mode:
            wind_scores = planner.calculate_wind_resource_score_vectorized()
            wave_scores = planner.calculate_wave_operational_score_fast()
            distance_scores = planner.calculate_distance_to_shore_score_fast()
        else:
            wind_scores = planner.calculate_wind_resource_score()
            wave_scores = planner.calculate_wave_operational_score()
            distance_scores = planner.calculate_distance_to_shore_score()
        
        # Environmental score (simplified)
        # env_scores = np.random.uniform(0.4, 0.8, len(planner.grid_gdf))
        env_scores = planner.calculate_environmental_sensitivity_score_for_windfarm()
        
        # Normalize weights
        total_weight = wind_resource_weight + environmental_weight + economic_weight + operational_weight
        weights = {
            'wind_resource': wind_resource_weight / total_weight,
            'environmental': environmental_weight / total_weight,
            'economic': economic_weight / total_weight,
            'operational': operational_weight / total_weight
        }
        
        # Create results dataframe
        grid_with_scores = planner.grid_gdf.copy()
        grid_with_scores['wind_resource_score'] = wind_scores
        grid_with_scores['wave_operational_score'] = wave_scores
        grid_with_scores['distance_to_shore_score'] = distance_scores
        grid_with_scores['environmental_score'] = env_scores
        
        # Calculate composite scores
        suitability_scores = (
            weights['wind_resource'] * wind_scores +
            weights['operational'] * wave_scores +
            weights['economic'] * distance_scores +
            weights['environmental'] * env_scores
        )
        
        grid_with_scores['suitability_score'] = suitability_scores
        
        # Store original constraints for reporting
        original_constraints = {
            'min_wind_resource': min_wind_resource,
            'max_wave_height': max_wave_height
        }
        
        # Adaptive constraint relaxation
        constraint_adjustments = []
        wave_score_threshold = max(0, 1 - max_wave_height / 10)
        
        # First attempt with original constraints
        suitable_sites = grid_with_scores[
            (grid_with_scores['wind_resource_score'] >= min_wind_resource) &
            (grid_with_scores['wave_operational_score'] >= wave_score_threshold)
        ].copy()
        
        # Check data distribution for debugging
        print(f"Score distributions:")
        print(f"Wind resource: min={wind_scores.min():.3f}, max={wind_scores.max():.3f}, mean={wind_scores.mean():.3f}")
        print(f"Wave operational: min={wave_scores.min():.3f}, max={wave_scores.max():.3f}, mean={wave_scores.mean():.3f}")
        print(f"Suitability: min={suitability_scores.min():.3f}, max={suitability_scores.max():.3f}, mean={suitability_scores.mean():.3f}")
        
        # Adaptive constraint relaxation if needed
        if len(suitable_sites) == 0 and adaptive_constraints:
            print("No sites found with original constraints, applying adaptive relaxation...")
            
            # Relax wind resource constraint
            wind_percentile_20 = np.percentile(wind_scores, 20)
            if min_wind_resource > wind_percentile_20:
                new_min_wind = max(0.1, wind_percentile_20)
                constraint_adjustments.append(f"Wind resource lowered from {min_wind_resource:.2f} to {new_min_wind:.2f}")
                min_wind_resource = new_min_wind
            
            # Relax wave constraint
            wave_percentile_80 = np.percentile(wave_scores, 80)
            new_max_wave = min(8.0, 10 * (1 - wave_percentile_80))  # Convert back to wave height
            if new_max_wave > max_wave_height:
                constraint_adjustments.append(f"Wave height increased from {max_wave_height:.1f}m to {new_max_wave:.1f}m")
                max_wave_height = new_max_wave
            
            # Recalculate with relaxed constraints
            wave_score_threshold = max(0, 1 - max_wave_height / 10)
            
            suitable_sites = grid_with_scores[
                (grid_with_scores['wind_resource_score'] >= min_wind_resource) &
                (grid_with_scores['wave_operational_score'] >= wave_score_threshold)
            ].copy()
            
            print(f"After constraint relaxation: {len(suitable_sites)} suitable sites found")
        
        # If still no sites, take top percentage regardless of constraints
        if len(suitable_sites) == 0:
            print("Still no sites found, selecting top 10% by suitability score...")
            top_10_percent = max(1, len(grid_with_scores) // 10)
            suitable_sites = grid_with_scores.nlargest(top_10_percent, 'suitability_score')
            constraint_adjustments.append("Selected top 10% of sites regardless of original constraints")
        
        # Get top candidates
        top_sites = suitable_sites.nlargest(num_sites, 'suitability_score')
        
        # Create visualization
        map_html = create_wind_farm_map(top_sites, suitable_sites, region, weights)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Generate comprehensive report
        report = generate_adaptive_wind_farm_report(
            top_sites, weights, region, goal, planner.data_availability,
            original_constraints, constraint_adjustments, 
            min_wind_resource, max_wave_height, fast_mode, total_time
        )
        
        result = {
            'report': report,
            'map_html': map_html,
            'region_analyzed': region,
            'total_suitable_sites': len(suitable_sites),
            'top_candidates': len(top_sites),
            'weights_applied': weights,
            'original_constraints': original_constraints,
            'final_constraints': {
                'min_wind_resource': min_wind_resource,
                'max_wave_height': max_wave_height
            },
            'constraint_adjustments': constraint_adjustments,
            'score_statistics': {
                'wind_resource': {
                    'min': float(wind_scores.min()),
                    'max': float(wind_scores.max()),
                    'mean': float(wind_scores.mean())
                },
                'wave_operational': {
                    'min': float(wave_scores.min()),
                    'max': float(wave_scores.max()),
                    'mean': float(wave_scores.mean())
                },
                'suitability': {
                    'min': float(suitability_scores.min()),
                    'max': float(suitability_scores.max()),
                    'mean': float(suitability_scores.mean())
                }
            },
            'performance_metrics': {
                'total_time_seconds': round(total_time, 1),
                'grid_cells_processed': len(planner.grid_gdf),
                'fast_mode_used': fast_mode
            },
            'data_quality': planner.data_availability
        }
        
        print(f"Analysis completed in {total_time:.1f}s with {len(top_sites)} sites")
        return json.dumps(result)
        
    except Exception as e:
        print(f"Error in adaptive planning: {e}")
        import traceback
        traceback.print_exc()
        
        return json.dumps({
            'error': f'Planning failed: {str(e)}',
            'region': region,
            'suggestion': 'Check data availability and regional boundaries'
        })


def generate_adaptive_wind_farm_report(top_sites: gpd.GeoDataFrame,
                                     weights: Dict[str, float],
                                     region: str,
                                     goal: str,
                                     data_availability: Dict[str, bool],
                                     original_constraints: Dict,
                                     constraint_adjustments: List[str],
                                     final_min_wind: float,
                                     final_max_wave: float,
                                     fast_mode: bool,
                                     total_time: float) -> str:
    """Generate report with constraint adjustment information."""
    
    report_lines = []
    
    report_lines.append(f"# ADAPTIVE WIND FARM PLANNING REPORT - {region.upper()}")
    report_lines.append("=" * 60)
    report_lines.append(f"**Analysis completed in {total_time:.1f} seconds**")
    report_lines.append(f"**Region:** {region.title()}")
    report_lines.append("")
    
    # Constraint adjustments section
    if constraint_adjustments:
        report_lines.append("## CONSTRAINT ADJUSTMENTS")
        report_lines.append("⚠️ **Original constraints were too restrictive. Adjustments made:**")
        for adjustment in constraint_adjustments:
            report_lines.append(f"- {adjustment}")
        report_lines.append("")
        
        report_lines.append("**Original vs Final Constraints:**")
        report_lines.append(f"- Wind Resource: {original_constraints['min_wind_resource']:.2f} → {final_min_wind:.2f}")
        report_lines.append(f"- Wave Height: {original_constraints['max_wave_height']:.1f}m → {final_max_wave:.1f}m")
        report_lines.append("")
    else:
        report_lines.append("## CONSTRAINT STATUS")
        report_lines.append("✅ **Original constraints were successfully applied**")
        report_lines.append("")
    
    # Data quality
    if data_availability['using_fallback']:
        report_lines.append("## DATA QUALITY WARNING")
        report_lines.append("⚠️ **Using synthetic data** - Copernicus Marine Service unavailable")
        report_lines.append("- Results are estimates based on climatological models")
        report_lines.append("- Validation with real measurements essential")
        report_lines.append("")
    
    # Results table
    report_lines.append("## TOP WIND FARM CANDIDATE SITES")
    report_lines.append("| Rank | Coordinates | Suitability | Wind | Waves | Distance |")
    report_lines.append("|------|-------------|-------------|------|-------|----------|")
    
    for idx, site in top_sites.iterrows():
        coords = f"{site['center_lat']:.1f}°N, {abs(site['center_lon']):.1f}°{'W' if site['center_lon'] < 0 else 'E'}"
        
        quality = ("🌟" if site['suitability_score'] > 0.7 else
                  "✅" if site['suitability_score'] > 0.5 else
                  "⚠️" if site['suitability_score'] > 0.3 else "❌")
        
        report_lines.append(
            f"| {idx+1} {quality} | {coords} | {site['suitability_score']:.3f} | "
            f"{site['wind_resource_score']:.3f} | {site['wave_operational_score']:.3f} | "
            f"{site['distance_to_shore_score']:.3f} |"
        )
    
    report_lines.append("")
    
    # Regional assessment
    avg_suitability = top_sites['suitability_score'].mean()
    avg_wind = top_sites['wind_resource_score'].mean()
    
    report_lines.append("## REGIONAL ASSESSMENT")
    report_lines.append(f"**{region.title()} Wind Farm Development Potential:**")
    
    if avg_suitability > 0.6 and avg_wind > 0.5:
        potential = "HIGH POTENTIAL"
        recommendation = "Excellent region for offshore wind development"
    elif avg_suitability > 0.4:
        potential = "MODERATE POTENTIAL"
        recommendation = "Good development opportunities with proper site selection"
    else:
        potential = "CHALLENGING CONDITIONS"
        recommendation = "Consider detailed feasibility studies before proceeding"
    
    report_lines.append(f"- **Overall Potential:** {potential}")
    report_lines.append(f"- **Recommendation:** {recommendation}")
    report_lines.append(f"- **Average Suitability Score:** {avg_suitability:.3f}")
    report_lines.append(f"- **Average Wind Resource:** {avg_wind:.3f}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## DEVELOPMENT RECOMMENDATIONS")
    
    if constraint_adjustments:
        report_lines.append("### Priority Actions (Due to Constraint Adjustments):")
        report_lines.append("1. **Validate assumptions** with current meteorological data")
        report_lines.append("2. **Conduct detailed resource assessment** at top sites")
        report_lines.append("3. **Review regional feasibility** given relaxed constraints")
        report_lines.append("")
    
    report_lines.append("### Standard Development Sequence:")
    report_lines.append("1. Extended wind measurement campaigns (12+ months)")
    report_lines.append("2. Environmental impact assessments")
    report_lines.append("3. Geotechnical and bathymetric surveys")
    report_lines.append("4. Grid connection studies")
    report_lines.append("5. Stakeholder engagement and permitting")
    
    if data_availability['using_fallback']:
        report_lines.append("")
        report_lines.append("### Critical Data Needs:")
        report_lines.append("- **Real oceanographic measurements** to replace synthetic data")
        report_lines.append("- **Current wind and wave monitoring** for validation")
        report_lines.append("- **Regional climate studies** for long-term projections")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Generated by Adaptive Wind Farm Site Planner*")
    
    if constraint_adjustments:
        report_lines.append("*⚠️ Note: Constraints were automatically adjusted - review carefully*")
    
    return "\n".join(report_lines)


# Add debug function for constraint analysis
def analyze_wind_farm_constraints(run_context: RunContext[DataSourceTracker],
                                region: str = "africa",
                                cell_size_km: float = 15.0) -> str:
    """
    Debug function to analyze score distributions and suggest optimal constraints.
    
    Args:
        region: Target region for analysis
        cell_size_km: Grid cell size
    
    Return:
        JSON with score distributions and constraint recommendations
    """
    
    try:
        # Quick analysis without full processing
        planner = GlobalWindFarmPlanner(region=region, cell_size_km=cell_size_km)
        planner.create_grid()
        
        # Calculate scores
        wind_scores = planner.calculate_wind_resource_score_vectorized()
        wave_scores = planner.calculate_wave_operational_score_fast()
        
        # Calculate percentiles
        wind_percentiles = {
            '10th': np.percentile(wind_scores, 10),
            '25th': np.percentile(wind_scores, 25),
            '50th': np.percentile(wind_scores, 50),
            '75th': np.percentile(wind_scores, 75),
            '90th': np.percentile(wind_scores, 90)
        }
        
        wave_percentiles = {
            '10th': np.percentile(wave_scores, 10),
            '25th': np.percentile(wave_scores, 25),
            '50th': np.percentile(wave_scores, 50),
            '75th': np.percentile(wave_scores, 75),
            '90th': np.percentile(wave_scores, 90)
        }
        
        # Suggest optimal constraints
        suggested_min_wind = wind_percentiles['25th']  # Exclude bottom 25%
        suggested_max_wave_height = 10 * (1 - wave_percentiles['75th'])  # Top 25% wave conditions
        
        result = {
            'region': region,
            'grid_cells_analyzed': len(planner.grid_gdf),
            'wind_resource_distribution': {
                'percentiles': wind_percentiles,
                'mean': float(wind_scores.mean()),
                'std': float(wind_scores.std())
            },
            'wave_operational_distribution': {
                'percentiles': wave_percentiles,
                'mean': float(wave_scores.mean()),
                'std': float(wave_scores.std())
            },
            'suggested_constraints': {
                'min_wind_resource': round(suggested_min_wind, 2),
                'max_wave_height': round(suggested_max_wave_height, 1),
                'rationale': f"Wind: 25th percentile, Wave: 75th percentile conditions"
            },
            'constraint_impact_estimates': {
                'conservative': {
                    'min_wind': round(wind_percentiles['50th'], 2),
                    'expected_sites': f"{int(len(planner.grid_gdf) * 0.5)} sites (~50%)"
                },
                'moderate': {
                    'min_wind': round(wind_percentiles['25th'], 2),
                    'expected_sites': f"{int(len(planner.grid_gdf) * 0.75)} sites (~75%)"
                },
                'permissive': {
                    'min_wind': round(wind_percentiles['10th'], 2),
                    'expected_sites': f"{int(len(planner.grid_gdf) * 0.9)} sites (~90%)"
                }
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            'error': f'Constraint analysis failed: {str(e)}',
            'region': region
        })


def get_location_bounds(run_context: RunContext[DataSourceTracker], 
                                location: str,
                                buffer_km: float = 50.0,
                                offshore_focus: bool = True) -> str:
    """
    Get geographical bounds for any location using Nominatim's built-in boundingbox feature.
    Much more efficient than the point-based approach since it uses actual geographic boundaries.
    
    Args:
        location: Location name (e.g., "North Sea", "UK", "Africa", "Mediterranean Sea") 
                 or coordinate string ("57.5, -2.0")
        buffer_km: Additional buffer around the location in kilometers (default: 50km)
        offshore_focus: If True, expand bounds toward offshore areas (default: True)
    
    Examples:
        - "Africa" -> Returns actual continental boundaries
        - "North Sea" -> Returns sea boundaries if available
        - "UK" -> Returns UK boundaries
        - "57.5, -2.0" -> Returns bounds around coordinates
    
    Return:
        JSON string with location bounds and metadata
    """
    
    try:
        print(f"Getting bounds for location: {location}")
        
        # Handle coordinate input first (fastest)
        if ',' in location and len(location.split(',')) == 2:
            return _get_bounds_from_coordinates(location, buffer_km, offshore_focus)
        
        # Strategy 1: Use Nominatim with boundingbox (IMPROVED APPROACH)
        bounds, location_info = _try_nominatim_with_boundingbox(location, buffer_km, offshore_focus)
        
        # Strategy 2: Fallback to predefined maritime regions
        if not bounds:
            bounds, location_info = _try_maritime_regions(location, buffer_km)
        
        # Strategy 3: Final fallback to point-based geocoding (original approach)
        if not bounds:
            bounds, location_info = _try_nominatim_point_based(location, buffer_km, offshore_focus)
        
        if not bounds:
            return json.dumps({
                'error': f'Could not find bounds for location: {location}',
                'suggestion': 'Try using coordinates (lat, lon) or a more specific location name',
                'examples': ['North Sea', 'UK', 'Mediterranean Sea', 'Africa', '57.5, -2.0']
            })
        
        # Add data source tracking
        run_context.deps.add_source(f"Geographic bounds for {location}")
        
        result = {
            'location_name': location_info.get('display_name', location),
            'bounds': bounds,
            'buffer_applied_km': buffer_km,
            'offshore_focus': offshore_focus,
            'area_info': location_info,
            'success': True,
            'coordinates_format': 'min_lat, max_lat, min_lon, max_lon',
            'usage_note': 'These bounds can be used directly with plan_global_wind_farm_sites'
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            'error': f'Failed to get location bounds: {str(e)}',
            'location': location,
            'success': False
        })

