import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import copernicusmarine
from typing import Dict, List, Tuple, Optional
import json
import folium
from datetime import datetime, timedelta
import warnings
from scipy.spatial import cKDTree

from grid_system import ExplorationGridSystem


class GlobalWindFarmPlanner(ExplorationGridSystem):
    """
    Performance-optimized wind farm planner with vectorized operations and data chunking.
    """
    
    def __init__(self, region: str = "africa", cell_size_km: float = 10.0, max_grid_cells: int = 5000):
        """
        Initialize optimized global wind farm planner.
        
        Args:
            region: Target region
            cell_size_km: Grid cell size in kilometers  
            max_grid_cells: Maximum grid cells to prevent memory issues
        """
        self.region = region
        self.max_grid_cells = max_grid_cells
        self.bounds = self._get_region_bounds(region)
        
        # Adjust cell size if grid would be too large
        estimated_cells = self._estimate_grid_size(cell_size_km)
        if estimated_cells > max_grid_cells:
            adjusted_size = self._calculate_optimal_cell_size()
            print(f"Adjusting cell size from {cell_size_km}km to {adjusted_size:.1f}km for performance")
            cell_size_km = adjusted_size
        
        super().__init__(cell_size_km=cell_size_km, study_bounds=self.bounds)
        
        copernicusmarine.login(
            username="tdangmanh",
            password="aA123456",
            force_overwrite=True)

        # Data cache
        self.wind_data = None
        self.wave_data = None
        self.data_availability = {
            'wind_current': False,
            'wave_historical': False,
            'using_fallback': False
        }
        
    def _estimate_grid_size(self, cell_size_km: float) -> int:
        """Estimate number of grid cells for given cell size."""
        lat_range = self.bounds['max_lat'] - self.bounds['min_lat']
        lon_range = self.bounds['max_lon'] - self.bounds['min_lon']
        cell_size_degrees = cell_size_km / 111.0
        
        n_lat = int(lat_range / cell_size_degrees)
        n_lon = int(lon_range / cell_size_degrees)
        
        # Estimate offshore cells (roughly 30% of total for most regions)
        return int(n_lat * n_lon * 0.3)
    
    def _calculate_optimal_cell_size(self) -> float:
        """Calculate optimal cell size to stay under max_grid_cells limit."""
        lat_range = self.bounds['max_lat'] - self.bounds['min_lat']
        lon_range = self.bounds['max_lon'] - self.bounds['min_lon']
        
        # Target grid dimensions
        target_cells = self.max_grid_cells / 0.3  # Account for land filtering
        target_dimension = int(np.sqrt(target_cells))
        
        optimal_lat_size = lat_range / target_dimension
        optimal_lon_size = lon_range / target_dimension
        optimal_size_degrees = max(optimal_lat_size, optimal_lon_size)
        
        return optimal_size_degrees * 111.0  # Convert to km
    
    def _get_region_bounds(self, region: str) -> Dict:
        """Get bounding box for different regions with performance-optimized sizes."""
        
        region_bounds = {
            "africa": {
                'min_lat': -35.0, 'max_lat': 37.0,
                'min_lon': -25.0, 'max_lon': 52.0
            },
            "europe": {
                'min_lat': 50.0, 'max_lat': 72.0,  # Focus on North Sea/Baltic
                'min_lon': -15.0, 'max_lon': 35.0
            },
            "asia": {
                'min_lat': 20.0, 'max_lat': 45.0,  # Focus on East Asia
                'min_lon': 100.0, 'max_lon': 150.0
            },
            "north_america_east": {
                'min_lat': 30.0, 'max_lat': 50.0,  # US East Coast
                'min_lon': -85.0, 'max_lon': -60.0
            },
            "north_america_west": {
                'min_lat': 30.0, 'max_lat': 50.0,  # US West Coast  
                'min_lon': -130.0, 'max_lon': -115.0
            }
        }
        
        return region_bounds.get(region, region_bounds["africa"])
    
    def load_copernicus_wind_data_optimized(self, months_back: int = 3) -> Optional[xr.Dataset]:
        """
        Load wind data with optimized spatial/temporal chunking.
        Reduced months_back for faster loading.
        """
        print(f"Loading optimized wind data for {self.region} ({months_back} months)...")
        
        try:
            end_date = datetime.now() - timedelta(days=30)
            start_date = end_date - timedelta(days=months_back * 30)
            
            # Load with reduced temporal resolution for speed
            self.wind_data = copernicusmarine.open_dataset(
                dataset_id='cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',
                minimum_longitude=self.bounds['min_lon'],
                maximum_longitude=self.bounds['max_lon'],
                minimum_latitude=self.bounds['min_lat'],
                maximum_latitude=self.bounds['max_lat'],
                start_datetime=start_date.strftime('%Y-%m-%d'),
                end_datetime=end_date.strftime('%Y-%m-%d')
            )
            
            # Downsample temporally for performance (daily averages instead of hourly)
            print("Downsampling to daily averages for performance...")
            self.wind_data = self.wind_data.resample(time='1D').mean()
            
            self.data_availability['wind_current'] = True
            print(f"Wind data loaded and downsampled: {self.wind_data.dims}")
            return self.wind_data
            
        except Exception as e:
            print(f"Wind data unavailable: {e}")
            print("Using optimized synthetic data...")
            self.data_availability['using_fallback'] = True
            return self._create_optimized_synthetic_wind_data()
    
    def _create_optimized_synthetic_wind_data(self) -> xr.Dataset:
        """Create lightweight synthetic wind data."""
        
        # Reduced resolution for speed
        lat_step = max(0.25, (self.bounds['max_lat'] - self.bounds['min_lat']) / 50)
        lon_step = max(0.25, (self.bounds['max_lon'] - self.bounds['min_lon']) / 50)
        
        lat_range = np.arange(self.bounds['min_lat'], self.bounds['max_lat'], lat_step)
        lon_range = np.arange(self.bounds['min_lon'], self.bounds['max_lon'], lon_step)
        
        # Only 30 days of data for speed
        time_range = pd.date_range('2023-01-01', periods=30, freq='D')
        
        print(f"Creating synthetic wind data: {len(lat_range)} x {len(lon_range)} x {len(time_range)}")
        
        # Vectorized wind pattern generation
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        
        # Regional patterns
        if self.region == "africa":
            base_u = -3 + 2*np.sin(np.radians(lon_grid)) + np.cos(np.radians(lat_grid))
            base_v = 2 + np.sin(np.radians(lat_grid))
        elif self.region == "europe":
            base_u = 8 + 2*np.cos(np.radians(lat_grid))
            base_v = 1 + np.sin(np.radians(lon_grid))
        else:
            base_u = 6 + np.sin(np.radians(lat_grid + lon_grid))
            base_v = 3 + np.cos(np.radians(lat_grid))
        
        # Expand to time dimension efficiently
        eastward_wind = np.broadcast_to(base_u[np.newaxis, :, :], (len(time_range), len(lat_range), len(lon_range)))
        northward_wind = np.broadcast_to(base_v[np.newaxis, :, :], (len(time_range), len(lat_range), len(lon_range)))
        
        # Add some random variation
        np.random.seed(42)
        eastward_wind = eastward_wind + np.random.normal(0, 1, eastward_wind.shape)
        northward_wind = northward_wind + np.random.normal(0, 0.8, northward_wind.shape)
        
        return xr.Dataset({
            'eastward_wind': (('time', 'latitude', 'longitude'), eastward_wind),
            'northward_wind': (('time', 'latitude', 'longitude'), northward_wind)
        }, coords={
            'time': time_range,
            'latitude': lat_range,
            'longitude': lon_range
        })
    
    def calculate_wind_resource_score_vectorized(self) -> np.ndarray:
        """Vectorized wind resource calculation using spatial interpolation."""
        print("Calculating wind resource scores (vectorized)...")
        
        if self.wind_data is None:
            self.load_copernicus_wind_data_optimized()
        
        # Calculate wind speed and statistics
        wind_speed = np.sqrt(
            self.wind_data['eastward_wind']**2 + self.wind_data['northward_wind']**2
        )
        
        wind_mean = wind_speed.mean(dim='time')
        wind_capacity = xr.where(
            (wind_speed >= 3) & (wind_speed <= 25),
            xr.where(wind_speed <= 12, (wind_speed / 12) ** 3, 1.0),
            0.0
        ).mean(dim='time')
        
        # Extract grid coordinates
        grid_lats = self.grid_gdf['center_lat'].values
        grid_lons = self.grid_gdf['center_lon'].values
        
        # Vectorized interpolation
        wind_mean_interp = wind_mean.interp(
            latitude=xr.DataArray(grid_lats, dims='points'),
            longitude=xr.DataArray(grid_lons, dims='points'),
            method='linear'
        ).values
        
        wind_capacity_interp = wind_capacity.interp(
            latitude=xr.DataArray(grid_lats, dims='points'),
            longitude=xr.DataArray(grid_lons, dims='points'),
            method='linear'
        ).values
        
        # Vectorized score calculation
        scores = np.where(
            wind_mean_interp > 25, 0.1,  # Too windy
            np.where(
                wind_mean_interp < 3, 0.0,  # Too calm
                wind_capacity_interp * 0.8 + (wind_mean_interp / 15) * 0.2
            )
        )
        
        # Handle NaN values
        scores = np.nan_to_num(scores, nan=0.0)
        
        self.scores['wind_resource'] = scores
        print(f"Wind resource scoring complete (vectorized). Best score: {scores.max():.3f}")
        return scores
    
    def calculate_wave_operational_score_fast(self) -> np.ndarray:
        """Fast wave scoring using simplified patterns."""
        print("Calculating wave operational scores (fast method)...")
        
        # Use simplified distance-based wave patterns instead of loading large datasets
        scores = np.zeros(len(self.grid_gdf))
        
        # Vectorized distance to coast calculation
        grid_coords = np.column_stack([
            self.grid_gdf['center_lat'].values,
            self.grid_gdf['center_lon'].values
        ])
        
        # Simplified coastal distance calculation
        coast_distances = self._vectorized_coast_distance(grid_coords)
        
        # Wave height model: generally increases with distance from coast and latitude
        for i, (lat, lon, coast_dist) in enumerate(zip(
            self.grid_gdf['center_lat'], 
            self.grid_gdf['center_lon'], 
            coast_distances
        )):
            # Simple wave height model
            base_wave_height = 1.5 + coast_dist * 0.02 + abs(lat) * 0.03
            
            # Regional adjustments
            if self.region == "africa":
                if lon < 10:  # Atlantic side
                    base_wave_height += 0.5
            elif self.region == "europe":
                if lat > 55:  # North Sea/Norwegian Sea
                    base_wave_height += 0.8
            
            # Convert to operational score (lower waves = higher score)
            if base_wave_height < 2:
                wave_score = 1.0
            elif base_wave_height < 4:
                wave_score = 1.0 - (base_wave_height - 2) / 2 * 0.6
            else:
                wave_score = 0.4 - min((base_wave_height - 4) / 4 * 0.3, 0.3)
            
            scores[i] = max(0, wave_score)
        
        self.scores['wave_operational'] = scores
        print(f"Wave operational scoring complete (fast). Best score: {scores.max():.3f}")
        return scores
    
    def _vectorized_coast_distance(self, grid_coords: np.ndarray) -> np.ndarray:
        """Vectorized distance to coast calculation."""
        
        # Define coastal reference points for each region
        if self.region == "africa":
            coast_points = np.array([
                [-10, 0], [-15, 10], [-10, 20], [0, 35],    # Atlantic
                [15, 35], [25, 30], [35, 20], [40, 0],     # Mediterranean/Red Sea
                [45, -10], [35, -25], [20, -35], [0, -30]  # Indian Ocean
            ])
        elif self.region == "europe":
            coast_points = np.array([
                [-10, 50], [-5, 55], [0, 60], [10, 65],    # Atlantic/North Sea
                [15, 60], [25, 55], [30, 65]               # Baltic
            ])
        else:
            # Default: region boundary points
            coast_points = np.array([
                [self.bounds['min_lon'], self.bounds['min_lat']],
                [self.bounds['max_lon'], self.bounds['min_lat']],
                [self.bounds['max_lon'], self.bounds['max_lat']],
                [self.bounds['min_lon'], self.bounds['max_lat']]
            ])
        
        # Build KDTree for fast nearest neighbor search
        coast_tree = cKDTree(coast_points)
        
        # Find nearest coast point for each grid cell
        distances, _ = coast_tree.query(grid_coords[:, [1, 0]])  # lon, lat order
        
        # Convert to km (rough)
        return distances * 111
    
    def calculate_distance_to_shore_score_fast(self) -> np.ndarray:
        """Fast distance scoring using pre-calculated coast distances."""
        print("Calculating distance to shore scores (fast)...")
        
        grid_coords = np.column_stack([
            self.grid_gdf['center_lat'].values,
            self.grid_gdf['center_lon'].values
        ])
        
        shore_distances = self._vectorized_coast_distance(grid_coords)
        
        # Vectorized distance scoring
        scores = np.where(
            (shore_distances >= 10) & (shore_distances <= 30), 1.0,  # Sweet spot
            np.where(
                (shore_distances >= 5) & (shore_distances < 10), 0.7,  # Close
                np.where(
                    (shore_distances > 30) & (shore_distances <= 100),
                    1.0 - (shore_distances - 30) / 70 * 0.5,  # Increasing costs
                    np.where(shore_distances > 100, 0.2, 0.1)  # Too far or too close
                )
            )
        )
        
        self.scores['distance_to_shore'] = scores
        print(f"Distance to shore scoring complete (fast). Best score: {scores.max():.3f}")
        return scores





def create_wind_farm_map(top_sites: gpd.GeoDataFrame,
                                  all_sites: gpd.GeoDataFrame,
                                  region: str,
                                  weights: Dict[str, float],
                                  max_background_points: int = 1000) -> str:
    """Create optimized map with limited background points for speed."""
    
    # Calculate center
    center_lat = (top_sites['center_lat'].min() + top_sites['center_lat'].max()) / 2
    center_lon = (top_sites['center_lon'].min() + top_sites['center_lon'].max()) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )
    
    # Sample background points for performance
    sample_size = min(max_background_points, len(all_sites))
    if sample_size < len(all_sites):
        sampled_sites = all_sites.sample(n=sample_size, random_state=42)
    else:
        sampled_sites = all_sites
    
    # Add background suitability
    for idx, site in sampled_sites.iterrows():
        suitability = site.get('suitability_score', 0)
        
        color = ('darkgreen' if suitability > 0.7 else
                'green' if suitability > 0.5 else
                'orange' if suitability > 0.3 else 'red')
        
        folium.CircleMarker(
            location=[site['center_lat'], site['center_lon']],
            radius=2,
            color=color,
            fill=True,
            fillOpacity=0.6,
            weight=0
        ).add_to(m)
    
    # Add top candidates
    for idx, site in top_sites.iterrows():
        folium.Marker(
            location=[site['center_lat'], site['center_lon']],
            popup=f"""
            <b>Wind Farm Site #{idx+1}</b><br>
            Suitability: {site['suitability_score']:.3f}<br>
            Wind: {site['wind_resource_score']:.3f}<br>
            Waves: {site['wave_operational_score']:.3f}
            """,
            icon=folium.Icon(color='green', icon='leaf')
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; height: 120px; 
                background-color: white; border:1px solid grey; z-index:9999; 
                font-size:11px; padding: 8px;">
    <h4 style="margin:0;">Wind Farm Sites - {region.title()}</h4>
    <p style="margin:2px 0;">🟢 Excellent (>0.7)</p>
    <p style="margin:2px 0;">🟡 Good (0.5-0.7)</p>
    <p style="margin:2px 0;">🟠 Fair (0.3-0.5)</p>
    <p style="margin:2px 0;">🔴 Poor (<0.3)</p>
    <p style="margin:2px 0;">🍃 Top {len(top_sites)} sites</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m.get_root().render()


def generate_wind_farm_report(top_sites: gpd.GeoDataFrame,
                                       weights: Dict[str, float],
                                       region: str,
                                       goal: str,
                                       data_availability: Dict[str, bool],
                                       min_wind_resource: float,
                                       max_wave_height: float,
                                       fast_mode: bool,
                                       total_time: float) -> str:
    """Generate optimized report with performance metrics."""
    
    report_lines = []
    
    report_lines.append(f"# OPTIMIZED WIND FARM PLANNING REPORT - {region.upper()}")
    report_lines.append("=" * 60)
    report_lines.append(f"**Analysis completed in {total_time:.1f} seconds**")
    report_lines.append(f"**Region:** {region.title()}")
    report_lines.append(f"**Mode:** {'Fast' if fast_mode else 'Standard'}")
    report_lines.append("")
    
    # Performance summary
    report_lines.append("## PERFORMANCE SUMMARY")
    report_lines.append(f"- **Total Processing Time:** {total_time:.1f} seconds")
    report_lines.append(f"- **Grid Cells Analyzed:** {len(top_sites)}")
    report_lines.append(f"- **Analysis Mode:** {'Fast algorithms' if fast_mode else 'Standard algorithms'}")
    if data_availability['using_fallback']:
        report_lines.append("- **Data Source:** Synthetic (Copernicus unavailable)")
    else:
        report_lines.append("- **Data Source:** Copernicus Marine Service")
    report_lines.append("")
    
    # Results table
    report_lines.append("## TOP WIND FARM SITES")
    report_lines.append("| Rank | Coordinates | Suitability | Wind | Operations |")
    report_lines.append("|------|-------------|-------------|------|------------|")
    
    for idx, site in top_sites.iterrows():
        coords = f"{site['center_lat']:.1f}°N, {abs(site['center_lon']):.1f}°{'W' if site['center_lon'] < 0 else 'E'}"
        report_lines.append(
            f"| {idx+1} | {coords} | {site['suitability_score']:.3f} | "
            f"{site['wind_resource_score']:.3f} | {site['wave_operational_score']:.3f} |"
        )
    
    report_lines.append("")
    
    # Summary recommendations
    avg_suitability = top_sites['suitability_score'].mean()
    
    report_lines.append("## DEVELOPMENT ASSESSMENT")
    if avg_suitability > 0.7:
        report_lines.append("**HIGH POTENTIAL** - Excellent conditions identified")
    elif avg_suitability > 0.5:
        report_lines.append("**MODERATE POTENTIAL** - Good development opportunities")
    else:
        report_lines.append("**LIMITED POTENTIAL** - Consider alternative approaches")
    
    report_lines.append("")
    report_lines.append("## NEXT STEPS")
    report_lines.append("1. Detailed resource assessment at top 3 sites")
    report_lines.append("2. Environmental impact evaluation")
    report_lines.append("3. Grid connection feasibility study")
    
    if fast_mode or data_availability['using_fallback']:
        report_lines.append("")
        report_lines.append("**Note:** This was a rapid assessment. Consider detailed analysis with:")
        report_lines.append("- Current oceanographic measurements")
        report_lines.append("- Higher resolution spatial analysis")
        report_lines.append("- Extended temporal datasets")
    
    return "\n".join(report_lines)




# class GlobalWindFarmPlanner(ExplorationGridSystem):
#     """
#     Extended exploration planner for global wind farm site selection using Copernicus marine data.
#     Handles data availability limitations and provides fallback options.
#     """
    
#     def __init__(self, region: str = "africa", cell_size_km: float = 10.0):
#         """
#         Initialize global wind farm planner.
        
#         Args:
#             region: Target region ("africa", "europe", "asia", "global", or custom bounds)
#             cell_size_km: Grid cell size in kilometers
#         """
#         self.region = region
#         self.bounds = self._get_region_bounds(region)
        
#         # Initialize parent class
#         super().__init__(cell_size_km=cell_size_km, study_bounds=self.bounds)
        
#         copernicusmarine.login(
#             username="tdangmanh",
#             password="aA123456",
#             force_overwrite=True)

#         # Copernicus data cache and availability flags
#         self.wind_data = None
#         self.wave_data = None
#         self.data_availability = {
#             'wind_current': False,
#             'wave_historical': False,
#             'using_fallback': False
#         }
        
#     def _get_region_bounds(self, region: str) -> Dict:
#         """Get bounding box for different regions."""
        
#         region_bounds = {
#             "africa": {
#                 'min_lat': -35.0, 'max_lat': 37.0,
#                 'min_lon': -25.0, 'max_lon': 52.0
#             },
#             "europe": {
#                 'min_lat': 35.0, 'max_lat': 72.0,
#                 'min_lon': -25.0, 'max_lon': 45.0
#             },
#             "asia": {
#                 'min_lat': -10.0, 'max_lat': 55.0,
#                 'min_lon': 60.0, 'max_lon': 180.0
#             },
#             "north_america": {
#                 'min_lat': 25.0, 'max_lat': 72.0,
#                 'min_lon': -180.0, 'max_lon': -50.0
#             },
#             "global": {
#                 'min_lat': -60.0, 'max_lat': 80.0,  # Avoid polar regions
#                 'min_lon': -180.0, 'max_lon': 180.0
#             }
#         }
        
#         return region_bounds.get(region, region_bounds["africa"])
    
#     def load_copernicus_wind_data(self, months_back: int = 6) -> Optional[xr.Dataset]:
#         """
#         Load current wind data from Copernicus Marine Service.
#         Uses recent data that should be available.
#         """
#         print(f"Loading current Copernicus wind data for {self.region}...")
        
#         try:
#             # Use recent but not current month to ensure data availability
#             end_date = datetime.now() - timedelta(days=30)  # 1 month back
#             start_date = end_date - timedelta(days=months_back * 30)
            
#             print(f"Requesting wind data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
#             # Load wind dataset with spatial and temporal subsetting
#             self.wind_data = copernicusmarine.open_dataset(
#                 dataset_id='cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',
#                 minimum_longitude=self.bounds['min_lon'],
#                 maximum_longitude=self.bounds['max_lon'],
#                 minimum_latitude=self.bounds['min_lat'],
#                 maximum_latitude=self.bounds['max_lat'],
#                 start_datetime=start_date.strftime('%Y-%m-%d'),
#                 end_datetime=end_date.strftime('%Y-%m-%d')
#             )
            
#             self.data_availability['wind_current'] = True
#             print(f"✅ Wind data loaded successfully: {self.wind_data.dims}")
#             return self.wind_data
            
#         except Exception as e:
#             print(f"⚠️ Current wind data unavailable: {e}")
#             print("🔄 Attempting to load historical wind data...")
#             return self._load_historical_wind_data()
    
#     def _load_historical_wind_data(self) -> Optional[xr.Dataset]:
#         """Load historical wind data as fallback."""
#         try:
#             # Try 2023 data as fallback
#             self.wind_data = copernicusmarine.open_dataset(
#                 dataset_id='cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',
#                 minimum_longitude=self.bounds['min_lon'],
#                 maximum_longitude=self.bounds['max_lon'],
#                 minimum_latitude=self.bounds['min_lat'],
#                 maximum_latitude=self.bounds['max_lat'],
#                 start_datetime='2023-01-01',
#                 end_datetime='2023-12-31'
#             )
            
#             print("✅ Using 2023 historical wind data")
#             return self.wind_data
            
#         except Exception as e:
#             print(f"⚠️ Historical wind data also unavailable: {e}")
#             print("🎲 Generating synthetic wind data for demonstration")
#             self.data_availability['using_fallback'] = True
#             return self._create_synthetic_wind_data()
    
#     def load_wave_data_with_fallback(self) -> Optional[xr.Dataset]:
#         """
#         Load wave data with known limitation (only until 2020).
#         Uses historical data and provides clear warnings about data vintage.
#         """
#         print(f"Loading historical wave data for {self.region} (2020 data - latest available)...")
        
#         try:
#             # Use 2020 data since that's the latest available
#             self.wave_data = copernicusmarine.open_dataset(
#                 dataset_id='cmems_obs-wave_glo_phy-swh_my_multi-l4-0.5deg_P1D-i',
#                 minimum_longitude=self.bounds['min_lon'],
#                 maximum_longitude=self.bounds['max_lon'],
#                 minimum_latitude=self.bounds['min_lat'],
#                 maximum_latitude=self.bounds['max_lat'],
#                 start_datetime='2020-01-01',
#                 end_datetime='2020-12-31'
#             )
            
#             self.data_availability['wave_historical'] = True
#             print("✅ 2020 wave data loaded (historical - may not represent current conditions)")
#             warnings.warn("Wave data is from 2020 - consider validation with recent measurements")
            
#             return self.wave_data
            
#         except Exception as e:
#             print(f"⚠️ Even 2020 wave data unavailable: {e}")
#             print("🎲 Using synthetic wave data based on regional climatology")
#             self.data_availability['using_fallback'] = True
#             return self._create_climatological_wave_data()
    
#     def _create_synthetic_wind_data(self) -> xr.Dataset:
#         """Create realistic synthetic wind data based on regional climatology."""
        
#         lat_range = np.arange(self.bounds['min_lat'], self.bounds['max_lat'], 0.125)
#         lon_range = np.arange(self.bounds['min_lon'], self.bounds['max_lon'], 0.125)
#         time_range = pd.date_range('2023-01-01', '2023-12-31', freq='6H')  # Reduce temporal resolution
        
#         print(f"Creating synthetic wind data: {len(lat_range)} x {len(lon_range)} x {len(time_range)}")
        
#         # Create realistic regional wind patterns
#         np.random.seed(42)
#         lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        
#         # Regional wind climatology patterns
#         if self.region == "africa":
#             # Trade winds in tropics, westerlies in south, Mediterranean patterns in north
#             base_u = np.where(lat_grid < 0, -3 + 2*np.sin(np.radians(lon_grid)), 
#                              np.where(lat_grid < 25, -5, 2))
#             base_v = np.where(lat_grid < 0, -2, np.where(lat_grid < 25, 2, -1))
#         elif self.region == "europe":
#             # Westerlies dominant
#             base_u = 6 + 2*np.cos(np.radians(lat_grid))
#             base_v = 1 + np.sin(np.radians(lon_grid))
#         else:
#             # Default pattern
#             base_u = 5 + 2*np.sin(np.radians(lat_grid))
#             base_v = 2 + np.cos(np.radians(lon_grid))
        
#         # Add seasonal and random variations
#         eastward_wind = np.zeros((len(time_range), len(lat_range), len(lon_range)))
#         northward_wind = np.zeros((len(time_range), len(lat_range), len(lon_range)))
        
#         for i, time in enumerate(time_range):
#             # Seasonal variation
#             seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * time.dayofyear / 365)
            
#             # Add realistic variability
#             u_var = np.random.normal(0, 2, (len(lat_range), len(lon_range)))
#             v_var = np.random.normal(0, 1.5, (len(lat_range), len(lon_range)))
            
#             eastward_wind[i] = (base_u * seasonal_factor + u_var)
#             northward_wind[i] = (base_v * seasonal_factor + v_var)
        
#         return xr.Dataset({
#             'eastward_wind': (('time', 'latitude', 'longitude'), eastward_wind),
#             'northward_wind': (('time', 'latitude', 'longitude'), northward_wind)
#         }, coords={
#             'time': time_range,
#             'latitude': lat_range,
#             'longitude': lon_range
#         })
    
#     def _create_climatological_wave_data(self) -> xr.Dataset:
#         """Create climatological wave data based on regional patterns."""
        
#         lat_range = np.arange(self.bounds['min_lat'], self.bounds['max_lat'], 0.5)
#         lon_range = np.arange(self.bounds['min_lon'], self.bounds['max_lon'], 0.5)
#         time_range = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
#         np.random.seed(42)
#         lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        
#         # Regional wave climatology
#         if self.region == "africa":
#             # Higher waves on Atlantic side, calmer in Mediterranean/Red Sea
#             base_wave = np.where(lon_grid < 10, 2.5 + 0.5*np.abs(lat_grid/30), 1.5)
#         elif self.region == "europe":
#             # North Sea and Atlantic have higher waves
#             base_wave = 2.0 + 0.3*lat_grid/60
#         else:
#             # Default pattern
#             base_wave = 2.0 + 0.5*np.abs(lat_grid)/30
        
#         # Add seasonal and random variations
#         wave_height = np.zeros((len(time_range), len(lat_range), len(lon_range)))
        
#         for i, time in enumerate(time_range):
#             # Winter storms (higher waves in winter months)
#             seasonal_factor = 1 + 0.4 * np.cos(2 * np.pi * (time.dayofyear - 15) / 365)
            
#             # Log-normal distribution for wave heights
#             wave_var = np.random.lognormal(0, 0.3, (len(lat_range), len(lon_range)))
#             wave_height[i] = base_wave * seasonal_factor * wave_var
        
#         return xr.Dataset({
#             'VAVH_INST': (('time', 'latitude', 'longitude'), wave_height)
#         }, coords={
#             'time': time_range,
#             'latitude': lat_range,
#             'longitude': lon_range
#         })
    
#     def calculate_wind_resource_score(self) -> np.ndarray:
#         """Calculate wind resource potential for each grid cell."""
#         print("Calculating wind resource scores...")
        
#         if self.wind_data is None:
#             self.load_copernicus_wind_data()
        
#         # Calculate wind speed magnitude
#         wind_speed = np.sqrt(
#             self.wind_data['eastward_wind']**2 + self.wind_data['northward_wind']**2
#         )
        
#         # Calculate temporal statistics
#         wind_mean = wind_speed.mean(dim='time')
#         wind_std = wind_speed.std(dim='time')
        
#         # Calculate capacity factor approximation
#         # Wind turbines typically have cut-in at 3 m/s, rated at 12-15 m/s, cut-out at 25 m/s
#         wind_capacity = xr.where(
#             (wind_speed >= 3) & (wind_speed <= 25),
#             xr.where(wind_speed <= 12, (wind_speed / 12) ** 3, 1.0),
#             0.0
#         ).mean(dim='time')
        
#         scores = np.zeros(len(self.grid_gdf))
        
#         for idx, cell in self.grid_gdf.iterrows():
#             lat = cell['center_lat']
#             lon = cell['center_lon']
            
#             try:
#                 # Find nearest grid point
#                 point_mean = wind_mean.sel(latitude=lat, longitude=lon, method='nearest').values
#                 point_capacity = wind_capacity.sel(latitude=lat, longitude=lon, method='nearest').values
#                 point_std = wind_std.sel(latitude=lat, longitude=lon, method='nearest').values
                
#                 # Combined score: mean wind, capacity factor, and consistency
#                 if point_mean > 25:  # Too high wind speeds
#                     resource_score = 0.1
#                 elif point_mean < 3:  # Too low wind speeds
#                     resource_score = 0.0
#                 else:
#                     # Weight capacity factor heavily, penalize high variability
#                     variability_factor = max(0, 1 - (point_std / max(point_mean, 1)) * 0.5)
#                     resource_score = point_capacity * 0.7 + (point_mean / 15) * 0.3
#                     resource_score *= variability_factor
                
#                 scores[idx] = max(0, min(1, resource_score))
                
#             except Exception as e:
#                 scores[idx] = 0.0
        
#         self.scores['wind_resource'] = scores
#         print(f"Wind resource scoring complete. Best score: {scores.max():.3f}")
#         return scores
    
#     def calculate_wave_operational_score(self) -> np.ndarray:
#         """Calculate wave operation suitability (lower waves = better operations)."""
#         print("Calculating wave operational scores...")
        
#         if self.wave_data is None:
#             self.load_wave_data_with_fallback()
        
#         wave_height = self.wave_data['VAVH_INST']
#         wave_mean = wave_height.mean(dim='time')
#         wave_95percentile = wave_height.quantile(0.95, dim='time')  # Extreme conditions
        
#         scores = np.zeros(len(self.grid_gdf))
        
#         for idx, cell in self.grid_gdf.iterrows():
#             lat = cell['center_lat']
#             lon = cell['center_lon']
            
#             try:
#                 mean_height = wave_mean.sel(latitude=lat, longitude=lon, method='nearest').values
#                 extreme_height = wave_95percentile.sel(latitude=lat, longitude=lon, method='nearest').values
                
#                 # Operational limits: <2m excellent, <4m good, >6m poor
#                 if mean_height < 1.5:
#                     wave_score = 1.0
#                 elif mean_height < 3.0:
#                     wave_score = 1.0 - (mean_height - 1.5) / 1.5 * 0.4
#                 elif mean_height < 5.0:
#                     wave_score = 0.6 - (mean_height - 3.0) / 2.0 * 0.4
#                 else:
#                     wave_score = 0.2
                
#                 # Penalty for extreme wave events (maintenance accessibility)
#                 if extreme_height > 8:
#                     wave_score *= 0.5
#                 elif extreme_height > 6:
#                     wave_score *= 0.7
                
#                 scores[idx] = max(0, wave_score)
                
#             except Exception as e:
#                 # Default moderate score for missing data
#                 scores[idx] = 0.5
        
#         self.scores['wave_operational'] = scores
#         print(f"Wave operational scoring complete. Best score: {scores.max():.3f}")
#         return scores
    
#     def calculate_distance_to_shore_score(self) -> np.ndarray:
#         """Calculate optimal distance to shore (economic vs technical tradeoff)."""
#         print("Calculating distance to shore scores...")
        
#         scores = np.zeros(len(self.grid_gdf))
        
#         for idx, cell in self.grid_gdf.iterrows():
#             lat = cell['center_lat']
#             lon = cell['center_lon']
            
#             # Estimate distance to nearest coastline (simplified)
#             shore_distance_km = self._estimate_distance_to_coast(lat, lon)
            
#             # Optimal distances: 10-50km (cable costs vs visual impact)
#             if 10 <= shore_distance_km <= 30:
#                 distance_score = 1.0  # Sweet spot
#             elif 5 <= shore_distance_km < 10:
#                 distance_score = 0.7  # Close to shore - visual/environmental concerns
#             elif 30 < shore_distance_km <= 100:
#                 distance_score = 1.0 - (shore_distance_km - 30) / 70 * 0.5  # Increasing cable costs
#             elif shore_distance_km > 100:
#                 distance_score = 0.2  # Very expensive cable runs
#             else:
#                 distance_score = 0.1  # Too close to shore
            
#             scores[idx] = distance_score
        
#         self.scores['distance_to_shore'] = scores
#         print(f"Distance to shore scoring complete. Best score: {scores.max():.3f}")
#         return scores
    
#     def _estimate_distance_to_coast(self, lat: float, lon: float) -> float:
#         """Estimate distance to nearest coastline using simplified coastal boundaries."""
        
#         # Regional coastal approximations
#         if self.region == "africa":
#             # Major African coastlines
#             atlantic_distance = abs(lon - (-10))  # Atlantic coast
#             mediterranean_distance = abs(lat - 35) if lat > 30 else 1000  # Mediterranean
#             indian_distance = abs(lon - 40) if lat < 0 else 1000  # Indian Ocean
#             red_sea_distance = abs(lon - 35) if 10 < lat < 30 else 1000  # Red Sea
            
#             coast_distance = min(atlantic_distance, mediterranean_distance, 
#                                indian_distance, red_sea_distance)
                               
#         elif self.region == "europe":
#             # European coastlines
#             atlantic_distance = abs(lon - (-10))
#             north_sea_distance = max(0, 6 - abs(lat - 56)) if -5 < lon < 10 else 1000
#             baltic_distance = max(0, 8 - abs(lat - 58)) if 10 < lon < 30 else 1000
#             mediterranean_distance = abs(lat - 40) if lat < 45 else 1000
            
#             coast_distance = min(atlantic_distance, north_sea_distance, 
#                                baltic_distance, mediterranean_distance)
#         else:
#             # Default: distance to region boundary
#             coast_distance = min(
#                 abs(lat - self.bounds['min_lat']),
#                 abs(lat - self.bounds['max_lat']),
#                 abs(lon - self.bounds['min_lon']),
#                 abs(lon - self.bounds['max_lon'])
#             )
        
#         return coast_distance * 111  # Convert degrees to km


# def create_wind_farm_map(top_sites: gpd.GeoDataFrame, 
#                         all_sites: gpd.GeoDataFrame,
#                         region: str,
#                         weights: Dict[str, float],
#                         data_availability: Dict[str, bool]) -> str:
#     """Create interactive map for global wind farm site selection with data quality indicators."""
    
#     # Calculate map center
#     if len(top_sites) > 0:
#         center_lat = (top_sites['center_lat'].min() + top_sites['center_lat'].max()) / 2
#         center_lon = (top_sites['center_lon'].min() + top_sites['center_lon'].max()) / 2
#     else:
#         center_lat = 0
#         center_lon = 0
    
#     # Create map
#     m = folium.Map(
#         location=[center_lat, center_lon],
#         zoom_start=5,
#         tiles="CartoDB positron"
#     )
    
#     # Add background suitability heatmap (sample for performance)
#     sample_size = min(2000, len(all_sites))
#     sampled_sites = all_sites.sample(n=sample_size, random_state=42)
    
#     for idx, site in sampled_sites.iterrows():
#         suitability = site.get('suitability_score', 0)
        
#         # Color coding for suitability
#         if suitability > 0.7:
#             color = 'darkgreen'
#             opacity = 0.8
#         elif suitability > 0.5:
#             color = 'green'
#             opacity = 0.6
#         elif suitability > 0.3:
#             color = 'orange'
#             opacity = 0.4
#         else:
#             color = 'red'
#             opacity = 0.3
        
#         folium.CircleMarker(
#             location=[site['center_lat'], site['center_lon']],
#             radius=2,
#             popup=f"Suitability: {suitability:.2f}",
#             color=color,
#             fill=True,
#             fillOpacity=opacity,
#             weight=0
#         ).add_to(m)
    
#     # Add top candidates with detailed popups
#     for idx, site in top_sites.iterrows():
#         folium.Marker(
#             location=[site['center_lat'], site['center_lon']],
#             popup=f"""
#             <div style='width: 250px;'>
#             <h4>🌊 Wind Farm Site #{idx+1}</h4>
#             <b>Overall Suitability:</b> {site['suitability_score']:.3f}<br>
#             <b>Wind Resource:</b> {site['wind_resource_score']:.3f}<br>
#             <b>Wave Conditions:</b> {site['wave_operational_score']:.3f}<br>
#             <b>Shore Distance:</b> {site['distance_to_shore_score']:.3f}<br>
#             <b>Environmental:</b> {site['environmental_score']:.3f}<br>
#             <br>
#             <small><i>Coordinates: {site['center_lat']:.2f}°N, {abs(site['center_lon']):.2f}°{'W' if site['center_lon'] < 0 else 'E'}</i></small>
#             </div>
#             """,
#             tooltip=f"Top Wind Farm Site #{idx+1}",
#             icon=folium.Icon(color='green', icon='leaf', prefix='fa')
#         ).add_to(m)
    
#     # Add data quality indicator
#     data_quality_color = 'red' if data_availability['using_fallback'] else 'orange' if data_availability['wave_historical'] else 'green'
    
#     # Enhanced legend with data quality information
#     legend_html = f'''
#     <div style="position: fixed; top: 10px; right: 10px; width: 280px; height: 200px; 
#                 background-color: white; border:2px solid grey; z-index:9999; 
#                 font-size:11px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
#     <h4 style="margin:0 0 8px 0; color:#2E8B57;">🌊 Global Wind Farm Planner - {region.title()}</h4>
    
#     <p style="margin:4px 0; font-weight:bold;">Site Suitability:</p>
#     <div style="display:flex; align-items:center; margin:2px 0;">
#         <span style="width:8px; height:8px; background:darkgreen; margin-right:4px; display:inline-block;"></span>
#         <span style="font-size:10px;">Excellent (>0.7)</span>
#     </div>
#     <div style="display:flex; align-items:center; margin:2px 0;">
#         <span style="width:8px; height:8px; background:green; margin-right:4px; display:inline-block;"></span>
#         <span style="font-size:10px;">Good (0.5-0.7)</span>
#     </div>
#     <div style="display:flex; align-items:center; margin:2px 0;">
#         <span style="width:8px; height:8px; background:orange; margin-right:4px; display:inline-block;"></span>
#         <span style="font-size:10px;">Fair (0.3-0.5)</span>
#     </div>
#     <div style="display:flex; align-items:center; margin:2px 0;">
#         <span style="width:8px; height:8px; background:red; margin-right:4px; display:inline-block;"></span>
#         <span style="font-size:10px;">Poor (<0.3)</span>
#     </div>
    
#     <p style="margin:6px 0 2px 0; font-weight:bold;">Top Candidates:</p>
#     <div style="display:flex; align-items:center; margin:1px 0;">
#         <span style="color:green;">🍃</span>
#         <span style="font-size:10px; margin-left:4px;">Selected Sites ({len(top_sites)})</span>
#     </div>
    
#     <p style="margin:6px 0 2px 0; font-weight:bold;">Data Quality:</p>
#     <div style="display:flex; align-items:center; margin:1px 0;">
#         <span style="width:10px; height:10px; background:{data_quality_color}; margin-right:4px; display:inline-block;"></span>
#         <span style="font-size:9px;">{'Current data' if not data_availability['using_fallback'] and not data_availability['wave_historical'] else 'Historical/Synthetic data'}</span>
#     </div>
    
#     <div style="margin:6px 0 0 0; font-size:8px; color:#666; text-align:center;">
#     Weights: Wind {weights['wind_resource']:.0%}, Env {weights['environmental']:.0%}, Econ {weights['economic']:.0%}, Ops {weights['operational']:.0%}
#     </div>
#     </div>
#     '''
#     m.get_root().html.add_child(folium.Element(legend_html))
    
#     return m.get_root().render()


# def generate_wind_farm_report(top_sites: gpd.GeoDataFrame,
#                              weights: Dict[str, float],
#                              region: str,
#                              goal: str,
#                              data_availability: Dict[str, bool],
#                              min_wind_resource: float,
#                              max_wave_height: float,
#                              min_distance_shore: float,
#                              max_distance_shore: float) -> str:
#     """Generate comprehensive wind farm planning report with data quality assessment."""
    
#     report_lines = []
    
#     report_lines.append(f"# GLOBAL OFFSHORE WIND FARM PLANNING REPORT")
#     report_lines.append(f"## Region: {region.upper()}")
#     report_lines.append("=" * 70)
#     report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
#     report_lines.append(f"**Target Region:** {region.title()}")
#     report_lines.append(f"**Planning Objective:** {goal}")
#     report_lines.append("")
    
#     # Data quality assessment
#     report_lines.append("## DATA QUALITY ASSESSMENT")
#     if data_availability['using_fallback']:
#         report_lines.append("⚠️ **Data Quality: LIMITED** - Using synthetic/fallback data")
#         report_lines.append("- Copernicus Marine Service data unavailable")
#         report_lines.append("- Results based on climatological models")
#         report_lines.append("- **Recommendation:** Validate with recent measurements before site selection")
#     elif data_availability['wave_historical']:
#         report_lines.append("🔶 **Data Quality: PARTIAL** - Mixed current and historical data")
#         report_lines.append("- Wind data: Current/recent measurements ✅")
#         report_lines.append("- Wave data: 2020 historical data (latest available) ⚠️")
#         report_lines.append("- **Recommendation:** Update wave analysis with recent buoy/satellite data")
#     else:
#         report_lines.append("✅ **Data Quality: CURRENT** - Using latest available data")
#         report_lines.append("- All oceanographic data from recent measurements")
    
#     report_lines.append("")
    
#     # Data sources
#     report_lines.append("## OCEANOGRAPHIC DATA SOURCES")
#     if not data_availability['using_fallback']:
#         report_lines.append("- **Wind Data:** Copernicus Marine Service - Global Ocean Wind and Stress")
#         report_lines.append("  * Spatial Resolution: 0.125° (~14 km)")
#         report_lines.append("  * Temporal Resolution: Hourly")
#         if data_availability['wave_historical']:
#             report_lines.append("- **Wave Data:** Copernicus Marine Service - Global Ocean Wave Height (2020)")
#             report_lines.append("  * Spatial Resolution: 0.5° (~55 km)")
#             report_lines.append("  * Temporal Resolution: Daily")
#             report_lines.append("  * ⚠️ Data vintage: 2020 (latest available)")
#     else:
#         report_lines.append("- **Wind Data:** Synthetic climatological model")
#         report_lines.append("- **Wave Data:** Regional climatological patterns")
#         report_lines.append("- ⚠️ Limited accuracy - validation essential")
    
#     report_lines.append("")
    
#     # Selection criteria
#     report_lines.append("## WIND FARM SITE SELECTION CRITERIA")
#     report_lines.append("**Selection Weights:**")
#     for criterion, weight in weights.items():
#         report_lines.append(f"- {criterion.replace('_', ' ').title()}: {weight:.1%}")
#     report_lines.append("")
    
#     report_lines.append("**Constraints Applied:**")
#     report_lines.append(f"- Minimum Wind Resource Score: {min_wind_resource:.2f}")
#     report_lines.append(f"- Maximum Mean Wave Height: {max_wave_height:.1f}m")
#     report_lines.append(f"- Distance from Shore: {min_distance_shore:.0f}-{max_distance_shore:.0f} km")
#     report_lines.append("")
    
#     # Results
#     report_lines.append("## TOP OFFSHORE WIND FARM CANDIDATE SITES")
#     report_lines.append("| Rank | Coordinates | Overall | Wind | Waves | Distance | Environment |")
#     report_lines.append("|------|-------------|---------|------|-------|----------|-------------|")
    
#     for idx, site in top_sites.iterrows():
#         coords = f"{site['center_lat']:.2f}°N, {abs(site['center_lon']):.2f}°{'W' if site['center_lon'] < 0 else 'E'}"
        
#         # Quality indicator
#         if site['suitability_score'] > 0.7:
#             quality = "🌟 EXCELLENT"
#         elif site['suitability_score'] > 0.5:
#             quality = "✅ GOOD"
#         elif site['suitability_score'] > 0.3:
#             quality = "⚠️ FAIR"
#         else:
#             quality = "❌ POOR"
            
#         report_lines.append(
#             f"| {idx+1} {quality} | {coords} | {site['suitability_score']:.3f} | "
#             f"{site['wind_resource_score']:.3f} | {site['wave_operational_score']:.3f} | "
#             f"{site['distance_to_shore_score']:.3f} | {site['environmental_score']:.3f} |"
#         )
    
#     report_lines.append("")
    
#     # Regional assessment
#     avg_wind = top_sites['wind_resource_score'].mean()
#     avg_waves = top_sites['wave_operational_score'].mean()
#     avg_distance = top_sites['distance_to_shore_score'].mean()
    
#     report_lines.append("## REGIONAL WIND FARM DEVELOPMENT POTENTIAL")
#     report_lines.append(f"**{region.title()} Analysis Summary:**")
#     report_lines.append(f"- Average Wind Resource Quality: {avg_wind:.3f}")
#     report_lines.append(f"- Average Wave Operation Conditions: {avg_waves:.3f}")
#     report_lines.append(f"- Average Distance Suitability: {avg_distance:.3f}")
#     report_lines.append("")
    
#     # Development recommendations
#     if avg_wind > 0.6 and avg_waves > 0.6:
#         potential_rating = "HIGH POTENTIAL"
#         recommendation = "Excellent conditions for offshore wind development"
#     elif avg_wind > 0.4 and avg_waves > 0.4:
#         potential_rating = "MODERATE POTENTIAL" 
#         recommendation = "Good development opportunities with proper planning"
#     else:
#         potential_rating = "LIMITED POTENTIAL"
#         recommendation = "Consider alternative renewable energy approaches"
    
#     report_lines.append(f"**Development Potential: {potential_rating}**")
#     report_lines.append(f"- {recommendation}")
#     report_lines.append("")
    
#     # Next steps
#     report_lines.append("## DEVELOPMENT RECOMMENDATIONS & NEXT STEPS")
    
#     if data_availability['using_fallback']:
#         report_lines.append("### CRITICAL - Data Validation Required:")
#         report_lines.append("1. **Obtain current oceanographic data** from recent measurements")
#         report_lines.append("2. **Deploy measurement buoys** at top candidate sites")
#         report_lines.append("3. **Validate synthetic model** against real-world conditions")
#         report_lines.append("")
    
#     report_lines.append("### Site Development Sequence:")
#     report_lines.append("1. **Detailed wind resource assessment** (12+ months of measurements)")
#     report_lines.append("2. **Comprehensive environmental impact assessment**")
#     report_lines.append("3. **Geotechnical surveys** (seafloor conditions, bathymetry)")
#     report_lines.append("4. **Grid connection feasibility studies**")
#     report_lines.append("5. **Stakeholder engagement** (fishing, shipping, environmental groups)")
#     report_lines.append("6. **Regulatory approval process** (environmental permits, maritime authorizations)")
    
#     if data_availability['wave_historical']:
#         report_lines.append("")
#         report_lines.append("### Wave Data Update Priority:")
#         report_lines.append("- **Update wave analysis** with post-2020 data when available")
#         report_lines.append("- **Cross-validate** with satellite altimetry and buoy measurements")
#         report_lines.append("- **Consider climate change impacts** on future wave conditions")
    
#     report_lines.append("")
    
#     # Footer with methodology
#     report_lines.append("---")
#     report_lines.append("**METHODOLOGY NOTES:**")
#     report_lines.append("- Wind resource scores based on turbine power curves and capacity factors")
#     report_lines.append("- Wave operational scores consider maintenance accessibility and turbine survivability")
#     report_lines.append("- Distance scores balance cable costs with visual/environmental impacts")
#     report_lines.append("- Environmental scores simplified - detailed EIA required for final selection")
#     report_lines.append("")
#     report_lines.append("*Report generated by Global Wind Farm Site Planner*")
#     report_lines.append("*This analysis provides initial screening - detailed surveys essential before development*")
    
#     if data_availability['using_fallback'] or data_availability['wave_historical']:
#         report_lines.append("*⚠️ Results limited by data availability - validation with current measurements critical*")
    
#     return "\n".join(report_lines)