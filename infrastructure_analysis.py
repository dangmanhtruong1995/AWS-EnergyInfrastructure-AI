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

from config import DATASET_LEGEND_DICT, DATASET_LIST, SCENARIOS,\
    WIND_FARM_SCENARIOS
from utils import calculate_distance, load_data_and_process, add_data_source


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


def generate_infrastructure_proximity_report(df_points: gpd.GeoDataFrame, 
                                           layer_1: str, layer_2: str, 
                                           max_distance: float,
                                           nearby_infrastructure: gpd.GeoDataFrame) -> str:
    """Generate detailed infrastructure proximity analysis report."""
    
    report_lines = []
    
    # Header
    report_lines.append(f"# INFRASTRUCTURE PROXIMITY ANALYSIS REPORT")
    report_lines.append(f"## {DATASET_LEGEND_DICT.get(layer_1, layer_1.title())} Near {DATASET_LEGEND_DICT.get(layer_2, layer_2.title())}")
    report_lines.append("=" * 70)
    report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**Search Radius:** {max_distance} kilometers")
    report_lines.append("")
    
    # Summary statistics
    total_found = len(df_points)
    within_1km = len(df_points[df_points['Score'] <= 1.0])
    within_5km = len(df_points[df_points['Score'] <= 5.0])
    infrastructure_count = len(nearby_infrastructure)
    
    report_lines.append("## ANALYSIS SUMMARY")
    report_lines.append(f"- **Total {layer_1.title()} Found:** {total_found}")
    report_lines.append(f"- **Within 1 km:** {within_1km} ({within_1km/total_found*100:.1f}%)")
    report_lines.append(f"- **Within 5 km:** {within_5km} ({within_5km/total_found*100:.1f}%)")
    report_lines.append(f"- **Associated {layer_2.title()}:** {infrastructure_count}")
    report_lines.append("")
    
    # Distance distribution
    if total_found > 0:
        min_distance = df_points['Score'].min()
        max_distance_found = df_points['Score'].max()
        avg_distance = df_points['Score'].mean()
        
        report_lines.append("## DISTANCE STATISTICS")
        report_lines.append(f"- **Closest Distance:** {min_distance:.2f} km")
        report_lines.append(f"- **Furthest Distance:** {max_distance_found:.2f} km")
        report_lines.append(f"- **Average Distance:** {avg_distance:.2f} km")
        report_lines.append("")
        
        # Proximity categories
        very_close = len(df_points[df_points['Score'] <= 1.0])
        close = len(df_points[(df_points['Score'] > 1.0) & (df_points['Score'] <= 3.0)])
        moderate = len(df_points[(df_points['Score'] > 3.0) & (df_points['Score'] <= 7.0)])
        distant = len(df_points[df_points['Score'] > 7.0])
        
        report_lines.append("## PROXIMITY CLASSIFICATION")
        report_lines.append(f"- **Very Close (≤1 km):** {very_close} {layer_1}s")
        report_lines.append(f"- **Close (1-3 km):** {close} {layer_1}s")
        report_lines.append(f"- **Moderate (3-7 km):** {moderate} {layer_1}s")
        report_lines.append(f"- **Distant (>7 km):** {distant} {layer_1}s")
        report_lines.append("")
    
    # Top 15 closest locations table
    report_lines.append("## CLOSEST LOCATIONS (Top 15)")
    report_lines.append("| Rank | Name | Distance (km) | Coordinates | Proximity |")
    report_lines.append("|------|------|---------------|-------------|-----------|")
    
    for idx, (_, row) in enumerate(df_points.head(15).iterrows(), 1):
        # Extract coordinates properly
        try:
            if hasattr(row['geometry'], 'y'):
                coord_str = f"{row['geometry'].y:.2f}°N, {abs(row['geometry'].x):.2f}°{'W' if row['geometry'].x < 0 else 'E'}"
            else:
                coord_str = "N/A"
        except:
            coord_str = "N/A"
        
        # Proximity indicator
        distance = row['Score']
        if distance <= 1.0:
            proximity = "🔴 Very Close"
        elif distance <= 3.0:
            proximity = "🟡 Close"
        elif distance <= 7.0:
            proximity = "🟢 Moderate"
        else:
            proximity = "🔵 Distant"
        
        # Clean up the name
        name = str(row.get('Name', 'Unknown')).replace('Name_left', '').strip()
        if not name or name == 'Unknown':
            name = f"{layer_1.title()}-{idx}"
        
        report_lines.append(
            f"| {idx} | {name} | {distance:.2f} | {coord_str} | {proximity} |"
        )
    
    report_lines.append("")
    
    # Strategic insights
    report_lines.append("## STRATEGIC INSIGHTS")
    
    if within_1km > 0:
        report_lines.append(f"**High Proximity Zone ({within_1km} locations):**")
        report_lines.append(f"- {within_1km} {layer_1}s are within 1 km of existing {layer_2}")
        report_lines.append("- These locations offer excellent connectivity opportunities")
        report_lines.append("- Consider coordination protocols for operational safety")
        report_lines.append("")
    
    if avg_distance < 3.0:
        report_lines.append("**Infrastructure Dense Region:**")
        report_lines.append(f"- Average distance of {avg_distance:.1f} km indicates high infrastructure density")
        report_lines.append("- Excellent for development projects requiring connectivity")
        report_lines.append("- Enhanced coordination may be required")
    elif avg_distance > 7.0:
        report_lines.append("**Infrastructure Sparse Region:**")
        report_lines.append(f"- Average distance of {avg_distance:.1f} km indicates lower infrastructure density")
        report_lines.append("- May require additional infrastructure investment")
        report_lines.append("- Consider phased development approach")
    else:
        report_lines.append("**Moderate Infrastructure Density:**")
        report_lines.append(f"- Average distance of {avg_distance:.1f} km shows balanced infrastructure distribution")
        report_lines.append("- Good foundation for future development")
    
    report_lines.append("")
    
    # Operational recommendations
    report_lines.append("## OPERATIONAL RECOMMENDATIONS")
    
    if very_close > 0:
        report_lines.append("**Immediate Coordination Required:**")
        report_lines.append(f"- {very_close} locations within 1 km require immediate stakeholder engagement")
        report_lines.append("- Establish coordination protocols with existing operators")
        report_lines.append("- Review safety and operational procedures")
        report_lines.append("")
    
    report_lines.append("**Development Priorities:**")
    if within_5km > total_found * 0.7:
        report_lines.append("- Focus on locations with existing infrastructure proximity")
        report_lines.append("- Leverage existing connectivity for cost efficiency")
    else:
        report_lines.append("- Consider infrastructure expansion requirements")
        report_lines.append("- Evaluate connectivity investment needs")
    
    report_lines.append("")
    report_lines.append("**Risk Management:**")
    report_lines.append("- Implement proximity monitoring systems")
    report_lines.append("- Establish emergency response protocols")
    report_lines.append("- Coordinate maintenance schedules to minimize conflicts")
    
    report_lines.append("")
    
    # Next steps
    report_lines.append("## RECOMMENDED NEXT STEPS")
    report_lines.append("1. **Stakeholder Engagement:** Contact operators of nearby infrastructure")
    report_lines.append("2. **Detailed Site Assessment:** Conduct surveys for top priority locations")
    report_lines.append("3. **Coordination Protocols:** Establish operational coordination agreements")
    report_lines.append("4. **Safety Assessment:** Review proximity-related safety requirements")
    report_lines.append("5. **Regulatory Compliance:** Ensure compliance with proximity regulations")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Analysis completed using geospatial proximity analysis*")
    report_lines.append(f"*Search radius: {max_distance} km | Total features analyzed: {total_found}*")
    report_lines.append("*Distances calculated using UTM projection for accuracy*")
    
    return "\n".join(report_lines)


def generate_within_operation_report(container_features: gpd.GeoDataFrame,
                                   contained_features: gpd.GeoDataFrame, 
                                   container_layer: str, 
                                   contained_layer: str) -> str:
    """Generate detailed within operation analysis report."""
    
    report_lines = []
    
    # Header
    report_lines.append(f"# SPATIAL CONTAINMENT ANALYSIS REPORT")
    report_lines.append(f"## {DATASET_LEGEND_DICT.get(contained_layer, contained_layer.title())} Within {DATASET_LEGEND_DICT.get(container_layer, container_layer.title())}")
    report_lines.append("=" * 70)
    report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**Analysis Type:** Spatial containment (within operation)")
    report_lines.append("")
    
    # Summary statistics
    total_containers = len(container_features)
    containers_with_content = len(container_features[container_features['Score'] > 0])
    total_contained = len(contained_features)
    
    report_lines.append("## ANALYSIS SUMMARY")
    report_lines.append(f"- **Total {container_layer.title()}:** {total_containers}")
    report_lines.append(f"- **{container_layer.title()} with {contained_layer.title()}:** {containers_with_content}")
    report_lines.append(f"- **Total {contained_layer.title()} Found:** {total_contained}")
    report_lines.append(f"- **Containment Rate:** {containers_with_content/total_containers*100:.1f}%")
    report_lines.append("")
    
    # Distribution analysis
    if total_containers > 0:
        max_contained = int(container_features['Score'].max())
        avg_contained = container_features['Score'].mean()
        
        report_lines.append("## DISTRIBUTION STATISTICS")
        report_lines.append(f"- **Maximum {contained_layer.title()} per {container_layer}:** {max_contained}")
        report_lines.append(f"- **Average {contained_layer.title()} per {container_layer}:** {avg_contained:.1f}")
        
        # Categorize containers by content density
        empty = len(container_features[container_features['Score'] == 0])
        low_density = len(container_features[(container_features['Score'] > 0) & (container_features['Score'] <= 2)])
        medium_density = len(container_features[(container_features['Score'] > 2) & (container_features['Score'] <= 5)])
        high_density = len(container_features[container_features['Score'] > 5])
        
        report_lines.append("")
        report_lines.append("## DENSITY CLASSIFICATION")
        report_lines.append(f"- **Empty (0 {contained_layer}):** {empty} {container_layer}s")
        report_lines.append(f"- **Low Density (1-2 {contained_layer}):** {low_density} {container_layer}s")
        report_lines.append(f"- **Medium Density (3-5 {contained_layer}):** {medium_density} {container_layer}s")
        report_lines.append(f"- **High Density (>5 {contained_layer}):** {high_density} {container_layer}s")
        report_lines.append("")
    
    # Top containers table
    active_containers = container_features[container_features['Score'] > 0].sort_values('Score', ascending=False)
    
    report_lines.append("## ACTIVE LOCATIONS (Containing Features)")
    report_lines.append("| Rank | Name | Count | Coordinates | Density Level |")
    report_lines.append("|------|------|-------|-------------|---------------|")
    
    for idx, (_, row) in enumerate(active_containers.head(15).iterrows(), 1):
        # Extract coordinates properly
        try:
            if hasattr(row['geometry'], 'centroid'):
                coords = row['geometry'].centroid
                coord_str = f"{coords.y:.2f}°N, {abs(coords.x):.2f}°{'W' if coords.x < 0 else 'E'}"
            elif hasattr(row['geometry'], 'y'):
                coord_str = f"{row['geometry'].y:.2f}°N, {abs(row['geometry'].x):.2f}°{'W' if row['geometry'].x < 0 else 'E'}"
            else:
                coord_str = "N/A"
        except:
            coord_str = "N/A"
        
        # Density indicator
        count = int(row['Score'])
        if count == 0:
            density = "⚪ Empty"
        elif count <= 2:
            density = "🟢 Low"
        elif count <= 5:
            density = "🟡 Medium"
        else:
            density = "🔴 High"
        
        # Clean up the name
        name = str(row.get('Name', 'Unknown')).strip()
        if not name or name == 'Unknown':
            name = f"{container_layer.title()}-{idx}"
        
        report_lines.append(
            f"| {idx} | {name} | {count} | {coord_str} | {density} |"
        )
    
    report_lines.append("")
    
    # Contained features details
    if total_contained > 0:
        report_lines.append("## CONTAINED FEATURES DETAILS")
        report_lines.append("| Feature Name | Container | Coordinates |")
        report_lines.append("|--------------|-----------|-------------|")
        
        for idx, (_, row) in enumerate(contained_features.head(20).iterrows(), 1):
            try:
                if hasattr(row['geometry'], 'y'):
                    coord_str = f"{row['geometry'].y:.2f}°N, {abs(row['geometry'].x):.2f}°{'W' if row['geometry'].x < 0 else 'E'}"
                else:
                    coord_str = "N/A"
            except:
                coord_str = "N/A"
            
            feature_name = str(row.get('Name_left', 'Unknown')).strip()
            container_name = str(row.get('Name_right', 'Unknown')).strip()
            
            report_lines.append(f"| {feature_name} | {container_name} | {coord_str} |")
    
    report_lines.append("")
    
    # Strategic insights
    report_lines.append("## STRATEGIC INSIGHTS")
    
    if containers_with_content > 0:
        containment_rate = containers_with_content / total_containers
        
        if containment_rate > 0.7:
            report_lines.append("**High Activity Region:**")
            report_lines.append(f"- {containment_rate:.1%} of {container_layer}s contain {contained_layer} features")
            report_lines.append("- Indicates concentrated activity in this region")
            report_lines.append("- Strong correlation between licensing and actual development")
        elif containment_rate > 0.3:
            report_lines.append("**Moderate Activity Region:**")
            report_lines.append(f"- {containment_rate:.1%} of {container_layer}s contain {contained_layer} features")
            report_lines.append("- Selective development pattern observed")
            report_lines.append("- Opportunities for additional development exist")
        else:
            report_lines.append("**Low Activity Region:**")
            report_lines.append(f"- Only {containment_rate:.1%} of {container_layer}s contain {contained_layer} features")
            report_lines.append("- Significant untapped potential")
            report_lines.append("- Consider factors limiting development")
        
        report_lines.append("")
    
    # Risk and opportunity assessment
    if contained_layer == "seismic" and container_layer == "licences":
        report_lines.append("## SEISMIC RISK ASSESSMENT")
        if high_density > 0:
            report_lines.append(f"**High Risk Areas ({high_density} blocks):**")
            report_lines.append("- Multiple seismic events detected within license boundaries")
            report_lines.append("- Enhanced monitoring and risk assessment recommended")
            report_lines.append("- Consider seismic-resistant infrastructure design")
        
        if containers_with_content > 0:
            report_lines.append(f"**Seismically Active Licenses:** {containers_with_content} out of {total_containers}")
            report_lines.append("- Regular seismic monitoring protocols recommended")
            report_lines.append("- Emergency response plans should be updated")
        
        report_lines.append("")
    
    # Operational recommendations
    report_lines.append("## OPERATIONAL RECOMMENDATIONS")
    
    if high_density > 0:
        report_lines.append("**High Density Areas:**")
        report_lines.append(f"- {high_density} {container_layer}s with concentrated {contained_layer} activity")
        report_lines.append("- Implement enhanced monitoring protocols")
        report_lines.append("- Consider specialized operational procedures")
        report_lines.append("")
    
    report_lines.append("**Development Strategy:**")
    if containment_rate > 0.5:
        report_lines.append("- Focus resources on active areas with proven activity")
        report_lines.append("- Leverage existing knowledge from active locations")
    else:
        report_lines.append("- Investigate factors limiting activity in licensed areas")
        report_lines.append("- Consider incentives for development in underutilized blocks")
    
    report_lines.append("")
    report_lines.append("**Monitoring Requirements:**")
    report_lines.append("- Establish regular monitoring of high-activity areas")
    report_lines.append("- Implement early warning systems where appropriate")
    report_lines.append("- Coordinate monitoring efforts across adjacent areas")
    
    report_lines.append("")
    
    # Next steps
    report_lines.append("## RECOMMENDED NEXT STEPS")
    report_lines.append("1. **Detailed Assessment:** Focus on high-density locations for comprehensive analysis")
    report_lines.append("2. **Risk Evaluation:** Conduct specific risk assessments for concentrated activity areas")
    report_lines.append("3. **Monitoring Enhancement:** Upgrade monitoring systems in active regions")
    report_lines.append("4. **Stakeholder Coordination:** Engage with operators in high-activity zones")
    report_lines.append("5. **Regulatory Review:** Ensure compliance with containment-specific regulations")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Analysis completed using spatial containment operations*")
    report_lines.append(f"*Total features analyzed: {total_containers} containers, {total_contained} contained features*")
    report_lines.append("*Spatial relationships determined using geometric containment analysis*")
    
    return "\n".join(report_lines)