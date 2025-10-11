import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from os.path import join as pjoin


class SeismicDrillingAnalyzer:
    """
    Analyzes seismic activity vs drilling density to identify areas with 
    high seismic survey density but low recent drilling activity.
    """
    
    def __init__(self, proximity_radius_km: float = 50.0, min_earthquake_count: int = 1):
        """
        Initialize the analyzer with configurable parameters.
        
        Args:
            proximity_radius_km: Radius in km to consider wells as "nearby"
            min_earthquake_count: Minimum earthquakes required for a region to be analyzed
        """
        self.proximity_radius = proximity_radius_km
        self.min_earthquake_count = min_earthquake_count
        self.uk_bounds = {
            'lat_min': 49.0, 'lat_max': 61.0,
            'lon_min': -8.0, 'lon_max': 3.0
        }
    
    def load_and_validate_data(self, earthquake_file: str, well_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV files and perform data quality assessment.
        
        Args:
            earthquake_file: Path to earthquake CSV file
            well_file: Path to well production CSV file
            
        Returns:
            Tuple of (earthquake_df, well_df) with validated data
        """
        print("Loading data files...")
        
        # Load earthquake data
        earthquake_df = pd.read_csv(earthquake_file)
        print(f"Loaded {len(earthquake_df)} earthquake records")
        
        # Load well data
        well_df = pd.read_csv(well_file)
        print(f"Loaded {len(well_df)} well records")
        
        # Clean and validate earthquake data
        earthquake_df = self._validate_coordinates(earthquake_df, "earthquake")
        
        # Clean and validate well data (with coordinate swap detection)
        well_df = self._validate_coordinates(well_df, "well")
        well_df = self._detect_and_fix_coordinate_swap(well_df)
        
        return earthquake_df, well_df
    
    def _validate_coordinates(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and clean coordinate data."""
        initial_count = len(df)
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=['Lat', 'Lon'])
        
        # Remove rows with invalid coordinates (non-numeric)
        df = df[pd.to_numeric(df['Lat'], errors='coerce').notna()]
        df = df[pd.to_numeric(df['Lon'], errors='coerce').notna()]
        
        # Convert to numeric
        df['Lat'] = pd.to_numeric(df['Lat'])
        df['Lon'] = pd.to_numeric(df['Lon'])
        
        final_count = len(df)
        if final_count < initial_count:
            print(f"Removed {initial_count - final_count} {data_type} records with invalid coordinates")
        
        return df
    
    def _detect_and_fix_coordinate_swap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect if Lat/Lon columns are swapped based on UK geographic bounds.
        """
        lat_in_bounds = ((df['Lat'] >= self.uk_bounds['lat_min']) & 
                        (df['Lat'] <= self.uk_bounds['lat_max'])).mean()
        
        lon_in_bounds = ((df['Lon'] >= self.uk_bounds['lon_min']) & 
                        (df['Lon'] <= self.uk_bounds['lon_max'])).mean()
        
        # Check if swapping would improve coordinate validity
        lat_as_lon = ((df['Lat'] >= self.uk_bounds['lon_min']) & 
                     (df['Lat'] <= self.uk_bounds['lon_max'])).mean()
        
        lon_as_lat = ((df['Lon'] >= self.uk_bounds['lat_min']) & 
                     (df['Lon'] <= self.uk_bounds['lat_max'])).mean()
        
        if (lat_as_lon > lat_in_bounds) and (lon_as_lat > lon_in_bounds):
            print("Detected coordinate swap - fixing...")
            df = df.copy()
            df['Lat'], df['Lon'] = df['Lon'].copy(), df['Lat'].copy()
            
        return df
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    
    def analyze_regional_distribution(self, earthquake_df: pd.DataFrame, well_df: pd.DataFrame) -> Dict:
        """
        Analyze earthquake and well distribution by region/field.
        """
        print("\nAnalyzing regional distribution...")
        
        # Count earthquakes by region
        earthquake_regions = earthquake_df['Region'].value_counts().to_dict()
        
        # Count wells by field
        well_fields = well_df['Field'].value_counts().to_dict()
        
        return {
            'earthquake_regions': earthquake_regions,
            'well_fields': well_fields,
            'top_earthquake_regions': dict(list(earthquake_regions.items())[:10]),
            'top_well_fields': dict(list(well_fields.items())[:10])
        }
    
    def analyze_spatial_proximity(self, earthquake_df: pd.DataFrame, well_df: pd.DataFrame) -> List[Dict]:
        """
        Analyze spatial relationship between earthquake regions and well locations.
        """
        print(f"\nAnalyzing spatial proximity (radius: {self.proximity_radius}km)...")
        
        # Group earthquakes by region
        earthquake_regions = earthquake_df.groupby('Region')
        
        region_analysis = []
        
        for region_name, region_group in earthquake_regions:
            if len(region_group) < self.min_earthquake_count:
                continue
                
            # Calculate regional centroid
            avg_lat = region_group['Lat'].mean()
            avg_lon = region_group['Lon'].mean()
            earthquake_count = len(region_group)
            
            # Count nearby wells
            nearby_wells = 0
            well_distances = []
            
            for _, well in well_df.iterrows():
                distance = self.calculate_distance(avg_lat, avg_lon, well['Lat'], well['Lon'])
                well_distances.append(distance)
                if distance <= self.proximity_radius:
                    nearby_wells += 1
            
            # Calculate ratio (avoid division by zero)
            ratio = earthquake_count / max(nearby_wells, 1)
            
            region_analysis.append({
                'region': region_name,
                'earthquake_count': earthquake_count,
                'nearby_wells': nearby_wells,
                'avg_lat': avg_lat,
                'avg_lon': avg_lon,
                'ratio': ratio,
                'min_well_distance': min(well_distances) if well_distances else float('inf')
            })
        
        # Sort by ratio (high seismic activity, low drilling density)
        region_analysis.sort(key=lambda x: x['ratio'], reverse=True)
        
        return region_analysis
    
    def create_grid_analysis(self, earthquake_df: pd.DataFrame, well_df: pd.DataFrame, 
                           cell_size_degrees: float = 0.5) -> List[Dict]:
        """
        Alternative grid-based analysis approach.
        """
        print(f"\nPerforming grid-based analysis (cell size: {cell_size_degrees}°)...")
        
        # Calculate bounds
        all_lats = list(earthquake_df['Lat']) + list(well_df['Lat'])
        all_lons = list(earthquake_df['Lon']) + list(well_df['Lon'])
        
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        
        # Create grid
        lat_cells = int(math.ceil((max_lat - min_lat) / cell_size_degrees))
        lon_cells = int(math.ceil((max_lon - min_lon) / cell_size_degrees))
        
        grid_analysis = []
        
        for i in range(lat_cells):
            for j in range(lon_cells):
                cell_min_lat = min_lat + i * cell_size_degrees
                cell_max_lat = min_lat + (i + 1) * cell_size_degrees
                cell_min_lon = min_lon + j * cell_size_degrees
                cell_max_lon = min_lon + (j + 1) * cell_size_degrees
                
                # Count earthquakes in cell
                eq_in_cell = earthquake_df[
                    (earthquake_df['Lat'] >= cell_min_lat) & 
                    (earthquake_df['Lat'] < cell_max_lat) &
                    (earthquake_df['Lon'] >= cell_min_lon) & 
                    (earthquake_df['Lon'] < cell_max_lon)
                ]
                
                # Count wells in cell
                wells_in_cell = well_df[
                    (well_df['Lat'] >= cell_min_lat) & 
                    (well_df['Lat'] < cell_max_lat) &
                    (well_df['Lon'] >= cell_min_lon) & 
                    (well_df['Lon'] < cell_max_lon)
                ]
                
                earthquake_count = len(eq_in_cell)
                well_count = len(wells_in_cell)
                
                if earthquake_count > 0 or well_count > 0:
                    ratio = earthquake_count / max(well_count, 1)
                    
                    grid_analysis.append({
                        'grid_id': f"{i}_{j}",
                        'lat_range': (cell_min_lat, cell_max_lat),
                        'lon_range': (cell_min_lon, cell_max_lon),
                        'center_lat': (cell_min_lat + cell_max_lat) / 2,
                        'center_lon': (cell_min_lon + cell_max_lon) / 2,
                        'earthquake_count': earthquake_count,
                        'well_count': well_count,
                        'ratio': ratio
                    })
        
        # Sort by ratio
        grid_analysis.sort(key=lambda x: x['ratio'], reverse=True)
        
        return grid_analysis
    
    def generate_summary_report(self, regional_analysis: List[Dict], 
                              grid_analysis: List[Dict], 
                              distribution_stats: Dict,
                              top_n: int = 15) -> Dict:
        """
        Generate comprehensive analysis summary.
        """
        print("\nGenerating summary report...")
        
        # Priority classification
        high_priority = [r for r in regional_analysis[:top_n] if r['ratio'] >= 4.0]
        medium_priority = [r for r in regional_analysis[:top_n] if 2.0 <= r['ratio'] < 4.0]
        low_priority = [r for r in regional_analysis[:top_n] if r['ratio'] < 2.0]
        
        report = {
            'analysis_parameters': {
                'proximity_radius_km': self.proximity_radius,
                'min_earthquake_count': self.min_earthquake_count
            },
            'data_summary': {
                'total_regions_analyzed': len(regional_analysis),
                'high_priority_areas': len(high_priority),
                'medium_priority_areas': len(medium_priority),
                'low_priority_areas': len(low_priority)
            },
            'top_regions': regional_analysis[:top_n],
            'priority_classification': {
                'high_priority': high_priority,
                'medium_priority': medium_priority,
                'low_priority': low_priority
            },
            'top_grid_cells': grid_analysis[:10],
            'distribution_stats': distribution_stats,
            'key_insights': self._generate_insights(regional_analysis, distribution_stats)
        }
        
        return report
    
    def _generate_insights(self, regional_analysis: List[Dict], distribution_stats: Dict) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # High seismic, zero drilling areas
        zero_drilling = [r for r in regional_analysis if r['nearby_wells'] == 0]
        if zero_drilling:
            insights.append(f"{len(zero_drilling)} regions have earthquake activity but zero wells within {self.proximity_radius}km")
        
        # Geographic patterns
        irish_sea_regions = [r for r in regional_analysis if 'IRISH SEA' in r['region']]
        if irish_sea_regions:
            insights.append(f"Irish Sea shows {irish_sea_regions[0]['earthquake_count']} earthquakes with {irish_sea_regions[0]['nearby_wells']} nearby wells")
        
        # Top earthquake region
        if regional_analysis:
            top_region = regional_analysis[0]
            insights.append(f"Highest priority area: {top_region['region']} ({top_region['earthquake_count']} earthquakes, {top_region['nearby_wells']} wells)")
        
        # Well concentration
        top_field = max(distribution_stats['well_fields'].items(), key=lambda x: x[1])
        insights.append(f"Most active drilling field: {top_field[0]} ({top_field[1]} wells)")
        
        return insights
    
    def save_results(self, report: Dict, output_file: str = "seismic_analysis_results.json"):
        """Save analysis results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    def run_complete_analysis(self, earthquake_file: str, well_file: str) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            earthquake_file: Path to earthquake CSV file
            well_file: Path to well production CSV file
            
        Returns:
            Complete analysis report dictionary
        """
        print("=== UK Seismic Activity vs Drilling Analysis ===")
        
        # Step 1: Load and validate data
        earthquake_df, well_df = self.load_and_validate_data(earthquake_file, well_file)
        
        # Step 2: Analyze regional distribution
        distribution_stats = self.analyze_regional_distribution(earthquake_df, well_df)
        
        # Step 3: Spatial proximity analysis
        regional_analysis = self.analyze_spatial_proximity(earthquake_df, well_df)
        
        # Step 4: Grid-based analysis (alternative approach)
        grid_analysis = self.create_grid_analysis(earthquake_df, well_df)
        
        # Step 5: Generate comprehensive report
        report = self.generate_summary_report(regional_analysis, grid_analysis, distribution_stats)
        
        # Step 6: Display results
        final_report = self._display_results(report)
        print(final_report)
        
        return report, final_report
    
    def _display_results(self, report: Dict):
        """Display formatted analysis results."""

        output = []
        output.append("\n" + "="*60)
        output.append("ANALYSIS RESULTS")
        output.append("="*60)
        
        output.append(f"\nData Summary:")
        output.append(f"- Total regions analyzed: {report['data_summary']['total_regions_analyzed']}")
        output.append(f"- High priority areas: {report['data_summary']['high_priority_areas']}")
        output.append(f"- Medium priority areas: {report['data_summary']['medium_priority_areas']}")
        
        output.append(f"\nTop 10 Priority Areas (High Seismic Activity, Low Drilling):")
        output.append("-" * 80)
        output.append(f"{'Rank':<4} {'Region':<20} {'Earthquakes':<11} {'Wells':<6} {'Ratio':<6} {'Location'}")
        output.append("-" * 80)
        
        for i, region in enumerate(report['top_regions'][:10], 1):
            output.append(f"{i:<4} {region['region'][:19]:<20} {region['earthquake_count']:<11} "
                  f"{region['nearby_wells']:<6} {region['ratio']:<6.1f} "
                  f"{region['avg_lat']:.3f}°N, {abs(region['avg_lon']):.3f}°W")
        
        output.append(f"\nKey Insights:")
        for insight in report['key_insights']:
            output.append(f"• {insight}")

        return "\n".join(output)


# Example usage and configuration
def main():
    """Example usage of the SeismicDrillingAnalyzer."""
    
    # Initialize analyzer with custom parameters
    analyzer = SeismicDrillingAnalyzer(
        proximity_radius_km=50.0,  # Consider wells within 50km as "nearby"
        min_earthquake_count=1     # Analyze regions with at least 1 earthquake
    )
    
    base_path = "/media/dangmanhtruong/147E655C7E65379E/TRUONG/Proposal_writing/Energy_Infrastructure_AI"
    well_data_path = pjoin(base_path, "datasets", "UKCS Daily Production Data", "UKCS_well_production_avg_data_processed.csv")
    seismic_data_path = pjoin(base_path, "datasets", "BGS_earthquake_data", "UK_BGS_earthquate_data.csv")   

    # Run complete analysis
    try:
        report, final_report = analyzer.run_complete_analysis(
            earthquake_file=seismic_data_path,
            well_file=well_data_path,
        )
        
        # Save results
        analyzer.save_results(report)
        
        # Additional custom analysis examples
        print("\n" + "="*60)
        print("CUSTOM ANALYSIS EXAMPLES")
        print("="*60)
        
        # Example: Change proximity radius
        analyzer_strict = SeismicDrillingAnalyzer(proximity_radius_km=25.0)
        print(f"\nWith stricter 25km radius:")
        # You could run partial analysis here
        
        # Example: Focus on high-activity regions only  
        analyzer_high_activity = SeismicDrillingAnalyzer(min_earthquake_count=3)
        print(f"\nFocusing on regions with 3+ earthquakes:")
        # You could run partial analysis here
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        print("Please ensure the CSV files are in the correct location.")
    except Exception as e:
        print(f"Analysis error: {e}")


if __name__ == "__main__":
    main()
