import os 
from os.path import join as pjoin
import boto3
import sys
import time

BASE_PATH = "/tmp"
S3_BUCKET = "oil-gas-datasets-436355390679"
s3_client = boto3.client('s3')


def get_dataset_from_s3(s3_key):
    """Download dataset from S3 to /tmp"""
    local_filename = s3_key.split('/')[-1]
    local_path = pjoin(BASE_PATH, local_filename)
    
    if not os.path.exists(local_path):
        start = time.time()
        print(f"    → Downloading {s3_key} from S3...", flush=True)
        sys.stdout.flush()

        s3_client.download_file(S3_BUCKET, s3_key, local_path)

        elapsed = time.time() - start
        print(f"    ✓ Downloaded in {elapsed:.2f}s", flush=True)
        sys.stdout.flush()
    else:
        print(f"    ✓ Using cached file: {local_path}", flush=True)
        sys.stdout.flush()
    
    return local_path


def get_shapefile_from_s3(dataset_name):
    """Download shapefile and all associated files"""
    base_key = DATASET_S3_KEYS[dataset_name].replace('.shp', '')
    base_name = base_key.split('/')[-1]
    
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
        try:
            s3_key = base_key + ext
            local_path = f"/tmp/{base_name}{ext}"
            if not os.path.exists(local_path):
                s3_client.download_file(S3_BUCKET, s3_key, local_path)
        except:
            pass  # Some extensions might not exist
    
    return f"/tmp/{base_name}.shp"

DATASET_S3_KEYS = {
    "seismic": "datasets/BGS_earthquake_data/UK_BGS_earthquate_data.csv",
    "drilling": "datasets/UKCS_Daily_Production_Data/UKCS_well_production_avg_data_processed.csv",
    "licences": "datasets/UKCS_OFF_WGS84/UKCS_Licensed_Blocks_WGS84.shp",
    "pipelines": "datasets/UKCS_OFF_WGS84/UKCS_Pipeline_Linear_WGS84.shp",
    "offshore_fields": "datasets/UKCS_OFF_WGS84/UKCS_Offshore_Fields_WGS84.shp",
    "wells": "datasets/UKCS_OFF_WGS84/UKCS_Wells_WGS84.shp",
    "windfarms": "datasets/EMODNet_windfarms/EMODnet_HA_Energy_WindFarms_pg_20250825.shp",
}

# Simple function to get local path
def get_dataset_path(dataset_name):
    s3_key = DATASET_S3_KEYS[dataset_name]
    return get_dataset_from_s3(s3_key)

# For backward compatibility
DATASETS = {k: lambda k=k: get_dataset_path(k) for k in DATASET_S3_KEYS.keys()}
DATASET_LIST = list(DATASET_S3_KEYS.keys())


DATASET_LEGEND_DICT = {
    "seismic": "Earthquake locations (2025)",
    "drilling": "Drilling locations",
    "licences": "UKCS licenced blocks area",
    "pipelines": "Pipelines",
    "offshore_fields": "Offshore field locations",
    "wells": "Well locations",
    "windfarms": "Active Offshore Wind Farms",
}

DATASET_SYMBOL_DICT = {
    "seismic": "Earthquake locations (2025)",
    "drilling": "Drilling locations",
    "licences": "UKCS licenced blocks area",
    "pipelines": "Pipelines",
    "offshore_fields": "Offshore field locations",
    "wells": "Well locations",
}

# Scenario modeling for MCDA
SCENARIOS = {    
    "balanced": {
        "safety": 0.25, "environment": 0.25, "technical": 0.25, "economic": 0.25
    },
    "economic_focus": {
        "safety": 0.2, "environment": 0.1, "technical": 0.2, "economic": 0.5
    },
    "safety_focus": {
        "safety": 0.5, "environment": 0.2, "technical": 0.2, "economic": 0.1
    },
    "technical_focus": {
        "safety": 0.1, "environment": 0.2, "technical": 0.5, "economic": 0.2
    },
    "environment_focus": {
        "safety": 0.1, "environment": 0.5, "technical": 0.2, "economic": 0.2
    }
}

# Wind farm specific scenarios
WIND_FARM_SCENARIOS = {
    "balanced_wind": {
        "wind_resource": 0.35, 
        "environmental": 0.25, 
        "economic": 0.25, 
        "operational": 0.15
    },
    "wind_resource_focus": {
        "wind_resource": 0.6, 
        "environmental": 0.15, 
        "economic": 0.15, 
        "operational": 0.1
    },
    "environmental_focus": {
        "wind_resource": 0.2, 
        "environmental": 0.5, 
        "economic": 0.15, 
        "operational": 0.15
    },
    "economic_focus": {
        "wind_resource": 0.25, 
        "environmental": 0.1, 
        "economic": 0.5, 
        "operational": 0.15
    },
    "operational_focus": {
        "wind_resource": 0.25, 
        "environmental": 0.15, 
        "economic": 0.2, 
        "operational": 0.4
    }
}

# WIND_FARM_SCENARIOS = {
#     "wind_resource_focus": {
#         "wind_resource": 0.6, "environmental": 0.2, "economic": 0.1, "wave_conditions": 0.1
#     },
#     "environmental_focus": {
#         "wind_resource": 0.2, "environmental": 0.5, "economic": 0.15, "wave_conditions": 0.15
#     },
#     "economic_focus": {
#         "wind_resource": 0.3, "environmental": 0.1, "economic": 0.4, "wave_conditions": 0.2
#     },
#     "balanced_wind_farm": {
#         "wind_resource": 0.35, "environmental": 0.25, "economic": 0.25, "wave_conditions": 0.15
#     }
# }

# Regional boundaries for global analysis
GLOBAL_REGIONS = {
    "africa": {
        "name": "Africa Continental Shelf",
        "bounds": {'min_lat': -35.0, 'max_lat': 37.0, 'min_lon': -25.0, 'max_lon': 52.0},
        "description": "Atlantic, Mediterranean, and Indian Ocean waters around Africa"
    },
    "europe": {
        "name": "European Waters", 
        "bounds": {'min_lat': 35.0, 'max_lat': 72.0, 'min_lon': -25.0, 'max_lon': 45.0},
        "description": "North Sea, Baltic Sea, Atlantic, and Mediterranean waters"
    },
    "asia_pacific": {
        "name": "Asia-Pacific Region",
        "bounds": {'min_lat': -10.0, 'max_lat': 55.0, 'min_lon': 60.0, 'max_lon': 180.0},
        "description": "Indian Ocean and Western Pacific waters"
    },
    "north_america": {
        "name": "North American Continental Shelf",
        "bounds": {'min_lat': 25.0, 'max_lat': 72.0, 'min_lon': -180.0, 'max_lon': -50.0},
        "description": "Atlantic and Pacific waters around North America"
    }
}
