from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Set, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import json
import contextily as ctx
import nltk
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

from config import BASE_PATH, DATASETS, DATASET_LIST, DATASET_LEGEND_DICT, SCENARIOS
from utils import calculate_distance, load_data_and_process


class MCDAEngine:
    def __init__(self):
        df_dict = {}
        layer_list = DATASET_LIST

        for layer_name in layer_list:
            df_dict[layer_name] = load_data_and_process(layer_name)

        self.df_dict = df_dict


    def get_objective_list(self):
        obj_list = ["safety", "economic", "technical", "environment"]
        return obj_list


    def calculate_objective(self, obj_name):
        if obj_name == "safety":
            obj_value = self.safety_objective()
        elif obj_name == "economic":
            obj_value = self.economic_objective()
        elif obj_name == "technical":
            obj_value = self.technical_objective()
        elif obj_name == "environment":
            obj_value = self.environment_objective()
        else:
            raise Exception(f"In class MCDAEngine, function calculate_objective: Undefined objective '{obj_name}.'")
        
        return obj_value


    def safety_objective(self):
        df_dict = self.df_dict

        n_licence = len(df_dict["licences"])
        n_well = len(df_dict["wells"])
        
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
        safety_obj_normalized = (safety_obj - np.min(safety_obj)) / (np.max(safety_obj) - np.min(safety_obj))

        return safety_obj_normalized


    def environment_objective(self):
        df_dict = self.df_dict

        n_licence = len(df_dict["licences"])
        n_well = len(df_dict["wells"])

        num_seismic_within_licence = np.zeros(n_licence)
        for licence_idx in range(n_licence):
            licence = df_dict["licences"]["geometry"].iloc[licence_idx]
            contains = licence.contains(df_dict["seismic"]["geometry"])
            num_seismic_within_licence[licence_idx] = np.sum(contains)
        env_obj = num_seismic_within_licence
        env_obj_normalized = (env_obj - np.min(env_obj)) / (np.max(env_obj) - np.min(env_obj))
        
        return env_obj_normalized


    def technical_objective(self):
        df_dict = self.df_dict

        n_licence = len(df_dict["licences"])
        n_well = len(df_dict["wells"]) 

        num_pipelines_going_through_licence = np.zeros(n_licence)
        for licence_idx in range(n_licence):
            licence = df_dict["licences"]["geometry"].iloc[licence_idx]
            intersects = licence.intersects(df_dict["pipelines"]["geometry"])
            num_pipelines_going_through_licence[licence_idx] = np.sum(intersects)

        # ADD: Windfarms calculation for technical feasibility
        num_windfarms_nearby_licence = np.zeros(n_licence)
        if "windfarms" in df_dict:
            for licence_idx in range(n_licence):
                licence = df_dict["licences"]["geometry"].iloc[licence_idx]
                # Check for windfarms within or intersecting the licence
                intersects = licence.intersects(df_dict["windfarms"]["geometry"])
                num_windfarms_nearby_licence[licence_idx] = np.sum(intersects)

        # Combine pipelines and windfarms for technical score
        tech_obj = num_pipelines_going_through_licence + num_windfarms_nearby_licence
        
        # Normalize and invert for technical (make all objectives higher is worse)
        tech_obj_normalized = (tech_obj - np.min(tech_obj)) / (np.max(tech_obj) - np.min(tech_obj))
        tech_obj_normalized = 1 - tech_obj_normalized

        return tech_obj_normalized
    

    def economic_objective(self):
        df_dict = self.df_dict

        n_licence = len(df_dict["licences"])
        n_well = len(df_dict["wells"]) 

        dist_to_nearest_field_km = np.zeros(n_licence)
        for licence_idx in range(n_licence):
            licence = df_dict["licences"]["geometry"].iloc[licence_idx]
            dist_min = np.min(licence.distance(df_dict["offshore_fields"]["geometry"]))
            dist_to_nearest_field_km[licence_idx] = dist_min
        econ_obj = dist_to_nearest_field_km
        econ_obj_normalized = (econ_obj - np.min(econ_obj)) / (np.max(econ_obj) - np.min(econ_obj))

        return econ_obj_normalized


def run_mcda(target:str, obj_1: str, obj_2: str, obj_3: str, obj_4: str,
            w_1: float, w_2: float, w_3: float, w_4: float):
    """ Do a Multi-Criterion Decision Analysis (MCDA) for the given target,
        ranking by a number of given objectives.
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
    Return:
        report: The final report in text form. Please note that for the objective scores, lower is better.
    """

    df_dict = {}
    layer_list = DATASET_LIST

    for layer_name in layer_list:
        df_dict[layer_name] = load_data_and_process(layer_name)

    mcda_engine = MCDAEngine()
    reference_obj_list = mcda_engine.get_objective_list()

    obj_list = [obj_1, obj_2, obj_3, obj_4]
    w_list = [w_1, w_2, w_3, w_4]
    obj_value_list = []
    print(f"Objectives ARE: {obj_list}")
    for obj_name  in obj_list:
        dist_list = [nltk.edit_distance(obj_name, elem) for elem in reference_obj_list]
        closest_obj_name = reference_obj_list[np.argmin(dist_list)]
        obj_value = mcda_engine.calculate_objective(closest_obj_name)
        obj_value_list.append(obj_value)

    score = w_list[0]*obj_value_list[0]
    for obj_idx, obj_value in enumerate(obj_value_list):
        if obj_idx == 0:
            continue
        score += w_list[obj_idx]*obj_value_list[obj_idx]

    df_licenses = df_dict["licences"].copy()
    df_licenses["Score"] = score.tolist()
    for obj_idx, obj_name in enumerate(obj_list):
        df_licenses[f"{obj_name}_score"] = obj_value_list[obj_idx].tolist()

    # Rank the licence blocks by score (higher is worse)
    df_rank = df_licenses.sort_values("Score").iloc[:20,:].copy()
    df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))

    df_rank["Coordinates"] = df_rank["geometry"].centroid
    df_rank = df_rank.drop('geometry', axis=1)
    report = df_rank.to_string(index=False)
    report = "REPORT OF 20 MOST RELEVANT POINTS FOUND DURING THE MULTI-CRITERIA DECISION ANALYSIS, ALONG WITH SCORES (lower score is better) \n\n" + report
    
    return report, df_rank


def normalize_weights(weights: dict) -> dict:
    """Ensure weights sum to 1."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("All weights are zero!")
    return {k: v / total for k, v in weights.items()}


def run_scenario_analysis(scenario_name: str, adjust: dict = None):
    """
    Run MCDA scenario analysis. Optionally adjust weights after choosing scenario.
    
    Args:
        scenario_name: Name of the scenario (e.g. 'economic_focus').
        adjust: Dictionary of adjustments (e.g. {'technical': +0.5, 'economic': -0.2}).
    """

    if scenario_name not in SCENARIOS:
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {list(SCENARIOS.keys())}")
    
    # Start from scenario
    weights = SCENARIOS[scenario_name].copy()

    # Apply adjustments
    if adjust:
        for k, v in adjust.items():
            if k not in weights:
                raise ValueError(f"Unknown weight '{k}'. Must be one of {list(weights.keys())}")
            weights[k] = max(0, weights[k] + v)  # prevent negatives
    
    # Normalize to sum = 1
    weights = normalize_weights(weights)

    # Run MCDA with updated weights
    report, df_rank = run_mcda(
        "licences", "safety", "environment", "technical", "economic",
        weights["safety"], weights["environment"],
        weights["technical"], weights["economic"]
    )

    return report, df_rank, weights


def main():
    report, df_rank, used_weights = run_scenario_analysis("economic_focus")

    print("Scenario used:", used_weights)
    print(report)

    print("")
    print(df_rank)

    print("")
    print("Used weights")
    print(used_weights)

    report, df_rank, used_weights = run_scenario_analysis("economic_focus", adjust={"technical": +0.5})

    print("Scenario used:", used_weights)
    print(report)

    print("")
    print(df_rank)

    print("")
    print("Used weights")
    print(used_weights)

    """
    w_1 = 0.9
    w_2 = 0.5
    w_3 = 0.3
    w_4 = 0.3
    report, df_rank = run_mcda(
        # "licences", "safety", "environment", "technical", "economic", 
        "licences", "economic", "technical", "safety", "environment", 
        w_1, w_2, w_3, w_4)
    """
    set_trace()

if __name__ == "__main__":
    main()