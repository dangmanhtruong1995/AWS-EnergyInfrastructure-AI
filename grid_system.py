import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import json
from typing import Dict, List, Tuple
from utils import load_data_and_process
from config import DATASET_LIST


class ExplorationGridSystem:
    """
    Creates and manages a grid system for exploration planning.
    Each grid cell gets scored on multiple criteria.
    """
    
    def __init__(self, cell_size_km: float = 5.0, study_bounds: Dict = None):
        """
        Initialize grid system.
        
        Args:
            cell_size_km: Size of each grid cell in kilometers
            study_bounds: Optional bounds dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        """
        self.cell_size_km = cell_size_km
        self.cell_size_degrees = cell_size_km / 111.0  # Rough conversion
        
        # Default UKCS bounds if not provided
        self.bounds = study_bounds or {
            'min_lat': 50.0, 'max_lat': 62.0,
            'min_lon': -8.0, 'max_lon': 3.0
        }
        
        self.grid_gdf = None
        self.scores = {}
        
    def create_grid(self) -> gpd.GeoDataFrame:
        """Create uniform grid over study area."""
        print(f"Creating {self.cell_size_km}km grid over UKCS...")
        
        # Calculate grid dimensions
        lat_range = self.bounds['max_lat'] - self.bounds['min_lat']
        lon_range = self.bounds['max_lon'] - self.bounds['min_lon']
        
        n_lat_cells = int(np.ceil(lat_range / self.cell_size_degrees))
        n_lon_cells = int(np.ceil(lon_range / self.cell_size_degrees))
        
        print(f"Grid dimensions: {n_lat_cells} x {n_lon_cells} = {n_lat_cells * n_lon_cells} cells")
        
        # Create grid cells
        grid_cells = []
        cell_id = 0
        
        for i in range(n_lat_cells):
            for j in range(n_lon_cells):
                # Cell bounds
                min_lat = self.bounds['min_lat'] + i * self.cell_size_degrees
                max_lat = min_lat + self.cell_size_degrees
                min_lon = self.bounds['min_lon'] + j * self.cell_size_degrees
                max_lon = min_lon + self.cell_size_degrees
                
                # Create polygon
                cell_polygon = Polygon([
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                    (min_lon, min_lat)
                ])
                
                # Calculate center point
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                
                grid_cells.append({
                    'cell_id': cell_id,
                    'grid_i': i,
                    'grid_j': j,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'geometry': cell_polygon
                })
                
                cell_id += 1
        
        # Create GeoDataFrame
        self.grid_gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        print(f"Created grid with {len(self.grid_gdf)} cells")
        
        return self.grid_gdf
    
    def calculate_seismic_score(self, radius_km: float = 25.0) -> np.ndarray:
        """
        Calculate seismic risk score for each grid cell.
        Higher score = more seismic events nearby = higher risk.
        """
        print("Calculating seismic scores...")
        
        df_seismic = load_data_and_process("seismic")
        
        # Convert to UTM for accurate distance calculation
        utm_crs = self.grid_gdf.estimate_utm_crs()
        grid_utm = self.grid_gdf.to_crs(utm_crs)
        seismic_utm = df_seismic.to_crs(utm_crs)
        
        scores = np.zeros(len(self.grid_gdf))
        
        for idx, cell in grid_utm.iterrows():
            # Create buffer around cell center
            print(f"Calculating ecological score for cell number {idx}")
            cell_center = cell.geometry.centroid
            buffer = cell_center.buffer(radius_km * 1000)  # Convert to meters
            
            # Count seismic events in buffer
            within_buffer = seismic_utm[seismic_utm.geometry.within(buffer)]
            scores[idx] = len(within_buffer)
        
        # Normalize to 0-1 (higher = worse for MCDA)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.scores['seismic'] = scores
        print(f"Seismic scoring complete. Max events per cell: {scores.max() * (df_seismic.shape[0] if scores.max() > 0 else 0):.0f}")
        
        return scores
    
    def calculate_ecological_sensitivity_score(self) -> np.ndarray:
        """
        Calculate ecological sensitivity score.
        For now, uses proximity to existing offshore fields as proxy.
        TODO: Replace with actual benthos/habitat data when available.
        """
        print("Calculating ecological sensitivity scores...")
        
        # Use offshore fields as proxy for sensitive areas
        df_fields = load_data_and_process("offshore_fields")
        
        utm_crs = self.grid_gdf.estimate_utm_crs()
        grid_utm = self.grid_gdf.to_crs(utm_crs)
        fields_utm = df_fields.to_crs(utm_crs)
        
        scores = np.zeros(len(self.grid_gdf))
        
        for idx, cell in grid_utm.iterrows():
            cell_center = cell.geometry.centroid
            
            # Calculate minimum distance to any offshore field
            min_distance = float('inf')
            for _, field in fields_utm.iterrows():
                distance = cell_center.distance(field.geometry)
                min_distance = min(min_distance, distance)
            
            # Convert distance to sensitivity score (closer = higher sensitivity)
            if min_distance != float('inf'):
                # Sensitivity decreases with distance (max sensitivity within 10km)
                sensitivity = max(0, 1 - (min_distance / (10 * 1000)))  # 10km threshold
                scores[idx] = sensitivity
        
        self.scores['ecological'] = scores
        print(f"Ecological sensitivity scoring complete. Max sensitivity: {scores.max():.3f}")
        
        return scores
    
    def calculate_infrastructure_proximity_score(self, radius_km: float = 50.0) -> np.ndarray:
        """
        Calculate infrastructure proximity score.
        Higher score = more infrastructure nearby = better for economics but worse for environment.
        """
        print("Calculating infrastructure proximity scores...")
        
        # Load infrastructure datasets
        infrastructure_types = ['wells', 'pipelines', 'offshore_fields']
        
        utm_crs = self.grid_gdf.estimate_utm_crs()
        grid_utm = self.grid_gdf.to_crs(utm_crs)
        
        scores = np.zeros(len(self.grid_gdf))
        
        for infra_type in infrastructure_types:
            df_infra = load_data_and_process(infra_type)
            infra_utm = df_infra.to_crs(utm_crs)
            
            for idx, cell in grid_utm.iterrows():
                cell_center = cell.geometry.centroid
                buffer = cell_center.buffer(radius_km * 1000)
                
                # Count infrastructure in buffer
                within_buffer = infra_utm[infra_utm.geometry.within(buffer)]
                scores[idx] += len(within_buffer)
        
        # Normalize to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.scores['infrastructure'] = scores
        print(f"Infrastructure proximity scoring complete. Max infrastructure count: {scores.max() * 100:.0f}")
        
        return scores
    
    def calculate_all_scores(self) -> Dict[str, np.ndarray]:
        """Calculate all scoring criteria."""
        # if self.grid_gdf is None:
        #     self.create_grid()
        
        # print("Calculating all grid scores...")
        
        # # Calculate all scores
        # self.calculate_seismic_score()
        # self.calculate_ecological_sensitivity_score()
        # self.calculate_infrastructure_proximity_score()
        
        # # Add scores to grid GeoDataFrame
        # for score_name, score_values in self.scores.items():
        #     self.grid_gdf[f'{score_name}_score'] = score_values
        
        # print("All scores calculated and added to grid")
        # return self.scores
        return self.calculate_all_scores_vectorized()
    
    def get_scored_grid(self) -> gpd.GeoDataFrame:
        """
        Get grid with all scores attached.
        FIXED: Ensures MCDA is run if suitability_score doesn't exist.
        """
        if not self.scores:
            self.calculate_all_scores_vectorized()
        
        # CRITICAL: Check if suitability_score exists, if not run MCDA
        if 'suitability_score' not in self.grid_gdf.columns:
            print("suitability_score not found, running MCDA...")
            self.run_mcda_analysis()
        
        return self.grid_gdf.copy()


    def calculate_seismic_score_optimized(self, radius_km: float = 25.0) -> np.ndarray:
        """
        Optimized seismic risk calculation using spatial indexing.
        ~100x faster than the buffer approach.
        """
        print("Calculating seismic scores (optimized)...")
        
        df_seismic = load_data_and_process("seismic")
        
        # Extract coordinates
        grid_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in self.grid_gdf.geometry])
        seismic_coords = np.array([[geom.x, geom.y] for geom in df_seismic.geometry])
        
        # Build spatial index
        seismic_tree = cKDTree(seismic_coords)
        
        # Convert radius to degrees (rough approximation)
        radius_degrees = radius_km / 111.0  # 1 degree ≈ 111 km
        
        scores = np.zeros(len(self.grid_gdf))
        
        # Vectorized distance calculation
        for idx, grid_point in enumerate(grid_coords):
            # Find all seismic events within radius
            indices = seismic_tree.query_ball_point(grid_point, radius_degrees)
            scores[idx] = len(indices)
            
            if idx % 1000 == 0:  # Progress indicator
                print(f"Processed {idx}/{len(grid_coords)} cells")
        
        # Normalize to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.scores['seismic'] = scores
        print(f"Seismic scoring complete. Max events per cell: {scores.max() * df_seismic.shape[0]:.0f}")
        
        return scores

    def calculate_ecological_sensitivity_score_optimized(self) -> np.ndarray:
        """
        Optimized ecological sensitivity calculation.
        """
        print("Calculating ecological sensitivity scores (optimized)...")
        
        df_fields = load_data_and_process("offshore_fields")
        
        # Extract centroids from grid
        grid_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in self.grid_gdf.geometry])
        
        # For polygon data (offshore fields), we'll use centroids for distance calculation
        field_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df_fields.geometry])
        
        # Build spatial index for fields
        field_tree = cKDTree(field_coords)
        
        scores = np.zeros(len(self.grid_gdf))
        
        # Calculate minimum distance to any field for each grid cell
        distances, _ = field_tree.query(grid_coords)
        
        # Convert distance to sensitivity score (closer = higher sensitivity)
        # Sensitivity decreases with distance (max sensitivity within 10km = ~0.09 degrees)
        max_sensitivity_distance = 0.09  # roughly 10km in degrees
        scores = np.maximum(0, 1 - (distances / max_sensitivity_distance))
        
        self.scores['ecological'] = scores
        print(f"Ecological sensitivity scoring complete. Max sensitivity: {scores.max():.3f}")
        
        return scores

    def calculate_infrastructure_proximity_score_optimized(self, radius_km: float = 50.0) -> np.ndarray:
        """
        Optimized infrastructure proximity calculation.
        """
        print("Calculating infrastructure proximity scores (optimized)...")
        
        infrastructure_types = ['wells', 'pipelines', 'offshore_fields']
        
        # Extract grid coordinates
        grid_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in self.grid_gdf.geometry])
        
        radius_degrees = radius_km / 111.0
        scores = np.zeros(len(self.grid_gdf))
        
        for infra_type in infrastructure_types:
            print(f"  Processing {infra_type}...")
            df_infra = load_data_and_process(infra_type)
            
            # Extract infrastructure coordinates
            if infra_type == 'pipelines':
                # For lines, use multiple points along the line
                infra_coords = []
                for geom in df_infra.geometry:
                    if hasattr(geom, 'coords'):
                        # LineString
                        infra_coords.extend([(x, y) for x, y in geom.coords])
                    elif hasattr(geom, 'geoms'):
                        # MultiLineString
                        for line in geom.geoms:
                            infra_coords.extend([(x, y) for x, y in line.coords])
                infra_coords = np.array(infra_coords)
            else:
                # For points and polygons, use centroids
                infra_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df_infra.geometry])
            
            if len(infra_coords) == 0:
                continue
                
            # Build spatial index
            infra_tree = cKDTree(infra_coords)
            
            # Count infrastructure within radius for each grid cell
            for idx, grid_point in enumerate(grid_coords):
                indices = infra_tree.query_ball_point(grid_point, radius_degrees)
                scores[idx] += len(indices)
        
        # Normalize to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.scores['infrastructure'] = scores
        print(f"Infrastructure proximity scoring complete. Max count: {scores.max() * 100:.0f}")
        
        return scores

    # Alternative even faster approach using GeoPandas spatial joins
    def calculate_all_scores_vectorized(self):
        """
        Ultra-fast vectorized approach using GeoPandas spatial operations.
        """
        print("Calculating all scores (vectorized approach)...")
        
        if self.grid_gdf is None:
            self.create_grid()
        
        # Convert to appropriate CRS for distance calculations
        utm_crs = self.grid_gdf.estimate_utm_crs()
        grid_utm = self.grid_gdf.to_crs(utm_crs)
        
        # Create buffers for all grid cells at once
        buffer_25km = grid_utm.copy()
        buffer_25km['geometry'] = buffer_25km.geometry.centroid.buffer(25000)  # 25km in meters
        
        buffer_50km = grid_utm.copy()
        buffer_50km['geometry'] = buffer_50km.geometry.centroid.buffer(50000)  # 50km in meters
        
        # Seismic scoring
        print("  Seismic scoring...")
        df_seismic = load_data_and_process("seismic")
        seismic_utm = df_seismic.to_crs(utm_crs)
        
        # Spatial join to count seismic events
        seismic_counts = gpd.sjoin(buffer_25km, seismic_utm, predicate='contains').groupby(level=0).size()
        seismic_scores = np.zeros(len(grid_utm))
        seismic_scores[seismic_counts.index] = seismic_counts.values
        seismic_scores = seismic_scores / seismic_scores.max() if seismic_scores.max() > 0 else seismic_scores
        
        # Infrastructure scoring
        print("  Infrastructure scoring...")
        infrastructure_types = ['wells', 'pipelines', 'offshore_fields']
        infra_scores = np.zeros(len(grid_utm))
        
        for infra_type in infrastructure_types:
            df_infra = load_data_and_process(infra_type)
            infra_utm = df_infra.to_crs(utm_crs)
            
            infra_counts = gpd.sjoin(buffer_50km, infra_utm, predicate='intersects').groupby(level=0).size()
            temp_scores = np.zeros(len(grid_utm))
            temp_scores[infra_counts.index] = infra_counts.values
            infra_scores += temp_scores
        
        infra_scores = infra_scores / infra_scores.max() if infra_scores.max() > 0 else infra_scores
        
        # Ecological scoring (using offshore fields as proxy)
        print("  Ecological scoring...")
        df_fields = load_data_and_process("offshore_fields")
        fields_utm = df_fields.to_crs(utm_crs)
        
        # Calculate minimum distance to any field
        grid_centroids = grid_utm.geometry.centroid
        ecological_scores = np.zeros(len(grid_utm))
        
        for idx, centroid in enumerate(grid_centroids):
            min_distance = fields_utm.geometry.distance(centroid).min()
            # Convert to sensitivity score (closer = higher sensitivity)
            ecological_scores[idx] = max(0, 1 - (min_distance / 10000))  # 10km threshold
        
        # Store scores
        self.scores = {
            'seismic': seismic_scores,
            'infrastructure': infra_scores,
            'ecological': ecological_scores
        }
        
        # Add to grid dataframe
        for score_name, score_values in self.scores.items():
            self.grid_gdf[f'{score_name}_score'] = score_values
                

        print("All scores calculated (vectorized)!")
        return self.scores


    def run_mcda_analysis(self, weights: Dict[str, float] = None) -> gpd.GeoDataFrame:
        """
        Run MCDA analysis on grid using calculated scores.
        FIXED: Ensures suitability_score is properly calculated and added.
        """
        if not self.scores:
            self.calculate_all_scores_vectorized()
        
        # Default weights (environment focused)
        if weights is None:
            weights = {
                'seismic': 0.4,      # 40% weight on seismic safety
                'ecological': 0.4,   # 40% weight on ecological sensitivity
                'infrastructure': 0.2 # 20% weight on infrastructure proximity
            }
        
        print(f"Running MCDA with weights: {weights}")
        
        # FIXED: Calculate composite suitability score using the stored scores
        seismic_weight = weights.get('seismic', 0.4)
        ecological_weight = weights.get('ecological', 0.4)
        infrastructure_weight = weights.get('infrastructure', 0.2)
        
        suitability_score = (
            seismic_weight * self.scores['seismic'] +
            ecological_weight * self.scores['ecological'] +
            infrastructure_weight * self.scores['infrastructure']
        )
        
        # CRITICAL FIX: Add suitability_score to the main grid dataframe
        self.grid_gdf['suitability_score'] = suitability_score
        
        # Rank cells (lower score = better suitability)
        self.grid_gdf['rank'] = self.grid_gdf['suitability_score'].rank(method='min')
        
        # Sort by rank
        ranked_grid = self.grid_gdf.sort_values('suitability_score').reset_index(drop=True)
        
        print(f"MCDA analysis complete. Best score: {suitability_score.min():.3f}, Worst: {suitability_score.max():.3f}")
        
        return ranked_grid