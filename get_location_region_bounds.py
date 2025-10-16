import json
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from pydantic_ai import RunContext
from schemas import DataSourceTracker



def _try_nominatim_with_boundingbox(location: str, buffer_km: float, offshore_focus: bool) -> Tuple[Optional[Dict], Dict]:
    """
    NEW IMPROVED METHOD: Use Nominatim's built-in boundingbox feature.
    This is much more accurate than calculating bounds from a center point.
    """
    try:
        geolocator = Nominatim(user_agent="energy_infrastructure_ai")
        
        # Try different search strategies for better results
        search_queries = [
            location,
            f"{location} region",
            f"{location} sea",
            f"{location} ocean",
            f"{location} coast"
        ]
        
        for query in search_queries:
            try:
                print(f"  Trying Nominatim query: '{query}'")
                
                # Get location with exactly_one=True for best result
                location_result = geolocator.geocode(query, exactly_one=True, timeout=10)
                
                if location_result and 'boundingbox' in location_result.raw:
                    # Extract boundingbox: [south, north, west, east]
                    bbox = location_result.raw['boundingbox']
                    south, north, west, east = map(float, bbox)
                    
                    print(f"  ✅ Found boundingbox: S={south}, N={north}, W={west}, E={east}")
                    
                    # Apply buffer if requested
                    if buffer_km > 0:
                        buffer_degrees = buffer_km / 111.0  # Rough km to degrees conversion
                        
                        # Apply offshore focus by expanding more toward likely offshore directions
                        if offshore_focus:
                            # For locations that are likely coastal/maritime, expand more seaward
                            if 'sea' in query.lower() or 'ocean' in query.lower() or 'coast' in query.lower():
                                # Expand more toward water (assume west/south for most cases)
                                west_buffer = buffer_degrees * 1.5
                                east_buffer = buffer_degrees * 0.8
                                south_buffer = buffer_degrees * 1.2
                                north_buffer = buffer_degrees * 0.8
                            else:
                                # Standard offshore expansion
                                west_buffer = east_buffer = buffer_degrees
                                south_buffer = north_buffer = buffer_degrees
                        else:
                            # Symmetric expansion
                            west_buffer = east_buffer = south_buffer = north_buffer = buffer_degrees
                        
                        # Apply buffers
                        south -= south_buffer
                        north += north_buffer
                        west -= west_buffer
                        east += east_buffer
                        
                        print(f"  Buffer applied ({buffer_km}km): S={south:.2f}, N={north:.2f}, W={west:.2f}, E={east:.2f}")
                    
                    # Convert to standard format
                    bounds = {
                        'min_lat': south,
                        'max_lat': north,
                        'min_lon': west,
                        'max_lon': east
                    }
                    
                    location_info = {
                        'display_name': location_result.address,
                        'center': {
                            'lat': location_result.latitude, 
                            'lon': location_result.longitude
                        },
                        'source': 'nominatim_boundingbox',
                        'search_query': query,
                        'original_boundingbox': bbox,
                        'osm_type': location_result.raw.get('osm_type', 'unknown'),
                        'place_id': location_result.raw.get('place_id', 'unknown')
                    }
                    
                    return bounds, location_info
                
                elif location_result:
                    print(f"  ⚠️ Location found but no boundingbox available")
                else:
                    print(f"  ❌ No location found for query: '{query}'")
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"  ⚠️ Geocoding error for '{query}': {e}")
                time.sleep(1)  # Rate limiting
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error for '{query}': {e}")
                continue
        
        return None, {}
        
    except Exception as e:
        print(f"❌ Nominatim boundingbox method failed: {e}")
        return None, {'error': str(e)}


def _try_nominatim_point_based(location: str, buffer_km: float, offshore_focus: bool) -> Tuple[Optional[Dict], Dict]:
    """
    FALLBACK METHOD: Original point-based approach when boundingbox is not available.
    """
    try:
        geolocator = Nominatim(user_agent="energy_infrastructure_ai")
        
        location_result = geolocator.geocode(location, exactly_one=True, timeout=10)
        
        if location_result:
            print(f"  Using point-based fallback for: {location_result.address}")
            
            # Calculate bounds around the point
            bounds = _calculate_bounds_from_point(
                location_result.latitude, 
                location_result.longitude, 
                buffer_km, 
                offshore_focus
            )
            
            location_info = {
                'display_name': location_result.address,
                'center': {
                    'lat': location_result.latitude, 
                    'lon': location_result.longitude
                },
                'source': 'nominatim_point_based_fallback',
                'note': 'No boundingbox available, calculated from center point'
            }
            
            return bounds, location_info
        
        return None, {}
        
    except Exception as e:
        return None, {'error': str(e)}


def _get_bounds_from_coordinates(coord_string: str, buffer_km: float, offshore_focus: bool) -> str:
    """Handle coordinate-based bounds calculation - unchanged from original."""
    try:
        parts = coord_string.split(',')
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid coordinates")
        
        bounds = _calculate_bounds_from_point(lat, lon, buffer_km, offshore_focus)
        
        location_info = {
            'display_name': f"Area around {lat:.2f}°N, {abs(lon):.2f}°{'W' if lon < 0 else 'E'}",
            'center': {'lat': lat, 'lon': lon},
            'source': 'coordinate_calculation'
        }
        
        result = {
            'location_name': location_info['display_name'],
            'bounds': bounds,
            'buffer_applied_km': buffer_km,
            'offshore_focus': offshore_focus,
            'area_info': location_info,
            'success': True
        }
        
        return json.dumps(result, indent=2)
        
    except (ValueError, IndexError) as e:
        return json.dumps({
            'error': f'Invalid coordinate format: {coord_string}',
            'expected_format': 'latitude, longitude (e.g., "57.5, -2.0")'
        })


def _try_maritime_regions(location: str, buffer_km: float) -> Tuple[Optional[Dict], Dict]:
    """Maritime regions fallback - unchanged from original but with improved coverage."""
    
    predefined_regions = {
        # Major Seas and Oceans
        'north sea': {
            'bounds': {'min_lat': 51.0, 'max_lat': 62.0, 'min_lon': -4.0, 'max_lon': 9.0},
            'description': 'North Sea between UK, Norway, Denmark, Germany, Netherlands, Belgium'
        },
        'baltic sea': {
            'bounds': {'min_lat': 53.0, 'max_lat': 66.0, 'min_lon': 10.0, 'max_lon': 30.0},
            'description': 'Baltic Sea between Scandinavia and Central Europe'
        },
        'mediterranean sea': {
            'bounds': {'min_lat': 30.0, 'max_lat': 46.0, 'min_lon': -6.0, 'max_lon': 36.0},
            'description': 'Mediterranean Sea between Europe, Africa, and Asia'
        },
        'black sea': {
            'bounds': {'min_lat': 40.0, 'max_lat': 48.0, 'min_lon': 27.0, 'max_lon': 42.0},
            'description': 'Black Sea between Europe and Asia'
        },
        'gulf of mexico': {
            'bounds': {'min_lat': 18.0, 'max_lat': 31.0, 'min_lon': -98.0, 'max_lon': -80.0},
            'description': 'Gulf of Mexico off US and Mexico coasts'
        },
        
        # Continents (rough offshore bounds)
        'africa': {
            'bounds': {'min_lat': -35.0, 'max_lat': 37.0, 'min_lon': -25.0, 'max_lon': 52.0},
            'description': 'Africa continental shelf and offshore waters'
        },
        'europe': {
            'bounds': {'min_lat': 35.0, 'max_lat': 72.0, 'min_lon': -25.0, 'max_lon': 45.0},
            'description': 'European waters including Atlantic, North Sea, Baltic, Mediterranean'
        },
        
        # Countries with significant offshore potential
        'uk': {
            'bounds': {'min_lat': 49.0, 'max_lat': 61.0, 'min_lon': -11.0, 'max_lon': 2.0},
            'description': 'United Kingdom with offshore waters'
        },
        'united kingdom': {
            'bounds': {'min_lat': 49.0, 'max_lat': 61.0, 'min_lon': -11.0, 'max_lon': 2.0},
            'description': 'United Kingdom with offshore waters'
        },
        'norway': {
            'bounds': {'min_lat': 57.0, 'max_lat': 72.0, 'min_lon': 4.0, 'max_lon': 32.0},
            'description': 'Norway with offshore waters'
        }
    }
    
    location_normalized = location.lower().strip()
    
    if location_normalized in predefined_regions:
        region_data = predefined_regions[location_normalized]
        bounds = region_data['bounds'].copy()
        
        # Apply buffer if specified
        if buffer_km > 0:
            buffer_degrees = buffer_km / 111.0
            bounds['min_lat'] -= buffer_degrees
            bounds['max_lat'] += buffer_degrees
            bounds['min_lon'] -= buffer_degrees
            bounds['max_lon'] += buffer_degrees
        
        location_info = {
            'display_name': region_data['description'],
            'source': 'predefined_maritime_regions',
            'original_bounds': region_data['bounds'],
            'buffered_bounds': bounds if buffer_km > 0 else None
        }
        
        return bounds, location_info
    
    return None, {}


def _calculate_bounds_from_point(lat: float, lon: float, buffer_km: float, offshore_focus: bool) -> Dict:
    """Calculate bounds around a central point - unchanged from original."""
    
    buffer_degrees = buffer_km / 111.0
    
    if offshore_focus:
        if 40 < lat < 70:  # Northern European waters
            west_expansion = buffer_degrees * 1.5
            east_expansion = buffer_degrees * 0.8
            north_expansion = buffer_degrees * 1.3
            south_expansion = buffer_degrees * 0.8
        else:
            west_expansion = east_expansion = buffer_degrees
            north_expansion = south_expansion = buffer_degrees
    else:
        west_expansion = east_expansion = buffer_degrees
        north_expansion = south_expansion = buffer_degrees
    
    bounds = {
        'min_lat': lat - south_expansion,
        'max_lat': lat + north_expansion,
        'min_lon': lon - west_expansion,
        'max_lon': lon + east_expansion
    }
    
    return bounds


# Example usage and testing function
# def test_improved_bounds_function():
#     """
#     Test function to demonstrate the improved boundingbox approach.
#     This would show how the function works with real examples.
#     """
    
#     # Mock run_context for testing
#     class MockRunContext:
#         class MockDeps:
#             def add_source(self, source): pass
#         deps = MockDeps()
    
#     mock_context = MockRunContext()
    
#     test_locations = [
#         "Africa",
#         "North Sea", 
#         "UK",
#         "Mediterranean Sea",
#         "57.5, -2.0"
#     ]
    
#     print("=== Testing Improved Location Bounds Function ===")
    
#     for location in test_locations:
#         print(f"\n--- Testing: {location} ---")
#         result = get_location_bounds(mock_context, location, buffer_km=100.0)
        
#         # Parse and display results
#         import json
#         data = json.loads(result)
        
#         if data.get('success'):
#             bounds = data['bounds']
#             print(f"✅ Success: {data['location_name']}")
#             print(f"   Bounds: {bounds['min_lat']:.2f}°S to {bounds['max_lat']:.2f}°N, {bounds['min_lon']:.2f}°W to {bounds['max_lon']:.2f}°E")
#             print(f"   Source: {data['area_info'].get('source', 'unknown')}")
#         else:
#             print(f"❌ Failed: {data.get('error', 'Unknown error')}")


# if __name__ == "__main__":
#     test_improved_bounds_function()
