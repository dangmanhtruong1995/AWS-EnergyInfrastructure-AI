import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
from os.path import join as pjoin
import math
import numpy as np
import sys
import time

from config import get_dataset_path, get_shapefile_from_s3, DATASET_S3_KEYS


def add_data_source(run_context, layer_list):
    for layer_name in layer_list:
        if layer_name == "seismic":
            run_context.deps.add_source("UK BGS earthquake data (https://www.earthquakes.bgs.ac.uk)")
        elif layer_name == "drilling":
            run_context.deps.add_source("UKCS daily production data")
        elif layer_name == "licences":
            run_context.deps.add_source("UKCS licensed blocks data (https://www.arcgis.com/home/item.html?id=92b08a672721407ca90ed26e67514af8)")
        elif layer_name == "wells":
            run_context.deps.add_source("UKCS wells data (https://www.arcgis.com/home/item.html?id=92b08a672721407ca90ed26e67514af8)")
        elif layer_name == "pipelines":
            run_context.deps.add_source("UKCS pipeline data (https://www.arcgis.com/home/item.html?id=92b08a672721407ca90ed26e67514af8)")
        elif layer_name == "offshore_fields":
            run_context.deps.add_source("UKCS offshore fields data (https://www.arcgis.com/home/item.html?id=92b08a672721407ca90ed26e67514af8)")
    


def load_data_and_process(layer_name: str):
    """
    Load and process datasets from S3.
    Handles both CSV and shapefile formats.
    """

    print(f"  → load_data_and_process({layer_name})", flush=True)
    sys.stdout.flush()
    
    start = time.time()

    # Determine file type from S3 key
    s3_key = DATASET_S3_KEYS[layer_name]

    print(f"  → S3 key: {s3_key}", flush=True)
    sys.stdout.flush()
    
    # Load data based on file type
    if s3_key.endswith('.csv'):
        print(f"  → Loading CSV...", flush=True)
        sys.stdout.flush()

        local_path = get_dataset_path(layer_name)

        print(f"  → Local path: {local_path}", flush=True)

        df = pd.read_csv(local_path)

        print(f"  → CSV loaded: {len(df)} rows", flush=True)
        sys.stdout.flush()
        
        # Convert to GeoDataFrame for CSV files with Lat/Lon
        if 'Lat' in df.columns and 'Lon' in df.columns:
            print(f"  → Converting to GeoDataFrame...", flush=True)
            sys.stdout.flush()

            df = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
                crs='EPSG:4326'
            )
    
    elif s3_key.endswith('.shp'):
        print(f"  → Loading shapefile...", flush=True)
        sys.stdout.flush()

        local_path = get_shapefile_from_s3(layer_name)

        print(f"  → Local path: {local_path}", flush=True)
        sys.stdout.flush()

        df = gpd.read_file(local_path)

        print(f"  → Shapefile loaded: {len(df)} rows", flush=True)
        sys.stdout.flush()
    
    else:
        raise ValueError(f"Unsupported file type for {layer_name}")
    
    print(f"  → Processing {layer_name}...", flush=True)
    sys.stdout.flush()

    # Process data based on layer type
    if layer_name == "seismic":
        df["Name"] = df["Region"] + ", " + df["Comment"]
        if 'geometry' not in df.columns:
            df["geometry"] = gpd.points_from_xy(df['Lon'], df['Lat'])
        df = df[["geometry", "Name"]]
    
    elif layer_name == "drilling":
        if 'geometry' not in df.columns:
            df["geometry"] = gpd.points_from_xy(df['Lon'], df['Lat'])
        df.rename(columns={'Field': 'Name'}, inplace=True)
        df = df[["geometry", "Name", "GasAvg"]]
    
    elif layer_name == "licences":
        df['Name'] = df['LICTYPE'] + df['LICNO'].astype(str) + '_' + df['BLOCKREF'] + '_' + df['BLOCKSUFFI']
        df = df[["geometry", "Name"]]
        df['Name'] = df['Name'].fillna("")
    
    elif layer_name == "pipelines":
        df.rename(columns={'PIPE_NAME': 'Name'}, inplace=True)
        df = df[["geometry", "Name"]]
    
    elif layer_name == "offshore_fields":
        df.rename(columns={'FIELDNAME': 'Name'}, inplace=True)
        df = df[["geometry", "Name"]]
    
    elif layer_name == "wells":
        df.rename(columns={'WELLREGNO': 'Name'}, inplace=True)
        df = df[["geometry", "Name", "ORIGINSTAT"]]
    
    # Ensure it's a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    elapsed = time.time() - start
    print(f"  ✓ load_data_and_process({layer_name}) complete in {elapsed:.2f}s", flush=True)
    sys.stdout.flush()

    return df


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c
