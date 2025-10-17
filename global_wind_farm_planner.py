import os
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
    
    def __init__(self, region: str = "africa", cell_size_km: float = 10.0,
                max_grid_cells: int = 5000, custom_bounds: Dict = None):
        """
        Initialize optimized global wind farm planner.
        
        Args:
            region: Target region
            cell_size_km: Grid cell size in kilometers  
            max_grid_cells: Maximum grid cells to prevent memory issues
            custom_bounds: Custom bounds (optional)
        """
        self.region = region
        self.max_grid_cells = max_grid_cells

        if custom_bounds:
            self.bounds = custom_bounds
        else:
            self.bounds = self._get_region_bounds(region)
        
        # Adjust cell size if grid would be too large
        estimated_cells = self._estimate_grid_size(cell_size_km)
        if estimated_cells > max_grid_cells:
            adjusted_size = self._calculate_optimal_cell_size()
            print(f"Adjusting cell size from {cell_size_km}km to {adjusted_size:.1f}km for performance")
            cell_size_km = adjusted_size
        
        super().__init__(cell_size_km=cell_size_km, study_bounds=self.bounds)
        
        # copernicusmarine.login(
        #     username="tdangmanh",
        #     password="aA123456",
        #     force_overwrite=True)

        copernicus_username = os.getenv('COPERNICUS_USERNAME')
        copernicus_password = os.getenv('COPERNICUS_PASSWORD')
        if copernicus_username and copernicus_password:
            copernicusmarine.login(
                username=copernicus_username,
                password=copernicus_password,
                force_overwrite=True
            )
            print("Copernicus Marine login successful")
        else:
            print("Copernicus credentials not found in environment variables")
            self.data_availability['using_fallback'] = True

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
    
    def calculate_environmental_sensitivity_score_for_windfarm(self) -> np.ndarray:
        """
        Calculate environmental sensitivity for wind farm development.
        Higher score = higher environmental sensitivity = worse for development.
        """
        print("Calculating environmental sensitivity scores for wind farms...")
        
        scores = np.zeros(len(self.grid_gdf))
        
        for idx, cell in self.grid_gdf.iterrows():
            lat = cell['center_lat']
            lon = cell['center_lon']
            
            # Base environmental sensitivity
            sensitivity = 0.3  # Base level
            
            # 1. Distance from coast (closer = higher sensitivity due to coastal ecosystems)
            # Estimate distance from coast using longitude (rough approximation)
            if self.region == "africa":
                # African coast references
                west_coast_dist = abs(lon - (-10))  # Rough Atlantic coast
                east_coast_dist = abs(lon - 40)     # Rough Indian Ocean coast
                coastal_dist = min(west_coast_dist, east_coast_dist)
            elif self.region == "europe":
                # European waters
                coastal_dist = abs(lon - 0)  # Distance from roughly European coast
            else:
                coastal_dist = 5  # Default assumption
            
            # Closer to coast = higher sensitivity
            coastal_sensitivity = max(0, 0.4 - (coastal_dist * 0.05))
            
            # 2. Latitude-based ecosystem sensitivity
            if self.region == "africa":
                if -5 <= lat <= 5:  # Equatorial waters - high marine biodiversity
                    ecosystem_sensitivity = 0.4
                elif lat < -20 or lat > 20:  # Temperate waters
                    ecosystem_sensitivity = 0.2
                else:  # Subtropical
                    ecosystem_sensitivity = 0.3
            elif self.region == "europe":
                if 54 <= lat <= 62:  # North Sea - important for bird migration
                    ecosystem_sensitivity = 0.3
                else:
                    ecosystem_sensitivity = 0.2
            else:
                ecosystem_sensitivity = 0.25
            
            # 3. Water depth sensitivity (estimated from distance from coast)
            estimated_depth = coastal_dist * 15  # Rough depth estimate
            if estimated_depth < 10:  # Too shallow
                depth_sensitivity = 0.3
            elif estimated_depth > 100:  # Too deep
                depth_sensitivity = 0.2
            else:  # Optimal depth range
                depth_sensitivity = 0.1
            
            # Combine all factors
            total_sensitivity = (
                sensitivity +
                coastal_sensitivity +
                ecosystem_sensitivity +
                depth_sensitivity
            )
            
            scores[idx] = min(1.0, total_sensitivity)
        
        print(f"Environmental sensitivity scoring complete. Average: {scores.mean():.3f}")
        return scores



def create_wind_farm_map(top_sites: gpd.GeoDataFrame,
                                  all_sites: gpd.GeoDataFrame,
                                  region: str,
                                  weights: Dict[str, float],
                                  max_background_points: int = 1000) -> str:
    """Create optimized map with limited background points for speed."""
    
    # Calculate center using ALL sites, not just top sites
    if len(all_sites) > 0:
        # Use bounds of all sites for proper regional coverage
        bounds = all_sites.bounds
        center_lat = (bounds.miny.min() + bounds.maxy.max()) / 2
        center_lon = (bounds.minx.min() + bounds.maxx.max()) / 2
        
        # Calculate zoom level based on full regional extent
        lat_range = bounds.maxy.max() - bounds.miny.min()
        lon_range = bounds.maxx.max() - bounds.minx.min()
        max_range = max(lat_range, lon_range)
        
        # Adjust zoom for larger regions
        if max_range > 40:  # Continental scale (like Africa)
            zoom_level = 3
        elif max_range > 20:  # Large regional scale
            zoom_level = 4
        elif max_range > 10:  # Regional scale
            zoom_level = 5
        elif max_range > 5:   # Sub-regional
            zoom_level = 6
        else:
            zoom_level = 7
            
    else:
        # Fallback - this shouldn't happen with valid data
        center_lat = (top_sites['center_lat'].min() + top_sites['center_lat'].max()) / 2
        center_lon = (top_sites['center_lon'].min() + top_sites['center_lon'].max()) / 2
        zoom_level = 5
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
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



