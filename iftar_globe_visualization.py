# Import required libraries
import datetime
import numpy as np
import rasterio
import time
import os
import logging
from rasterio.warp import transform_bounds
import math

# For visualization
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('iftar_globe.log')
    ]
)
logger = logging.getLogger('iftar_globe')

# Hardcoded paths to raster files
HIGH_RES_POPULATION_RASTER = "script/raster/Total_Muslim_Fasting_Population.tif"
LOW_RES_POPULATION_RASTER = "script/raster/tot_pop_mus_fast_low_res.tif"

# Cache for loaded raster data
HIGH_RES_CACHE = None
LOW_RES_CACHE = None
CHOROPLETH_DATA_CACHE = None
CARTESIAN_COORDS_CACHE = None  # New cache for pre-calculated cartesian coordinates

def initialize_app_data():
    """Load all data once at startup"""
    global HIGH_RES_CACHE, LOW_RES_CACHE, CHOROPLETH_DATA_CACHE, CARTESIAN_COORDS_CACHE
    
    logger.info("Loading raster data at application startup (one-time load)")
    
    # Load high-resolution data
    HIGH_RES_CACHE = load_population_raster(HIGH_RES_POPULATION_RASTER, is_high_res=True, force_reload=True)
    logger.info("High-resolution data loaded and cached successfully")
    
    # Load low-resolution data
    try:
        LOW_RES_CACHE = load_population_raster(LOW_RES_POPULATION_RASTER, is_high_res=False, force_reload=True)
        logger.info("Low-resolution data loaded and cached successfully")
        
        # Pre-calculate choropleth data
        CHOROPLETH_DATA_CACHE = prepare_choropleth_data(LOW_RES_CACHE)
        logger.info("Choropleth data prepared and cached successfully")
        
        # Pre-calculate cartesian coordinates for WebGL rendering
        if CHOROPLETH_DATA_CACHE:
            logger.info("Pre-calculating cartesian coordinates for WebGL rendering...")
            CARTESIAN_COORDS_CACHE = pre_calculate_cartesian_coords(CHOROPLETH_DATA_CACHE)
            logger.info(f"Cartesian coordinates pre-calculated for {len(CARTESIAN_COORDS_CACHE['x'])} points")
    except Exception as e:
        logger.error(f"Error loading low-resolution data: {str(e)}")
        logger.warning("Continuing without low-resolution data")
    
    logger.info("Data initialization complete")

def calculate_iftar_longitudes(current_time_minutes, window_minutes=10):
    """
    Calculate the longitude band for the iftar zone based on time
    
    Args:
        current_time_minutes: Current time in minutes since midnight (UTC)
        window_minutes: Width of the iftar window in minutes
    
    Returns:
        east_longitude: The eastern edge of the iftar band (earlier sunset)
        west_longitude: The western edge of the iftar band (later sunset)
    """
    # Earth rotates 15 degrees per hour or 0.25 degrees per minute
    degrees_per_minute = 0.25
    
    # Calculate the center longitude where the sun is setting
    # At noon (720 minutes), the sun is at the meridian (0 degrees longitude)
    # Convert time to equivalent longitude
    center_longitude = ((720 - current_time_minutes) * degrees_per_minute) % 360
    if center_longitude > 180:
        center_longitude -= 360  # Convert to -180 to 180 range
    
    # Calculate band edges
    # The total window width in degrees is window_minutes * degrees_per_minute
    # Half of this is added/subtracted to get the east/west longitudes
    window_half_width = (window_minutes * degrees_per_minute) / 2
    
    # Eastern edge is center plus half window (we add because we're going east with earlier sunset)
    east_longitude = center_longitude + window_half_width
    if east_longitude > 180:
        east_longitude -= 360
        
    # Western edge is center minus half window (we subtract because we're going west with later sunset)
    west_longitude = center_longitude - window_half_width
    if west_longitude < -180:
        west_longitude += 360
    
    logger.info(f"Calculated Iftar longitudes: east={east_longitude:.2f}°, west={west_longitude:.2f}°")
    return east_longitude, west_longitude

def load_population_raster(raster_path, is_high_res=True, force_reload=False):
    """
    Load population data from a GeoTIFF file, with caching
    
    Args:
        raster_path: Path to the population raster file
        is_high_res: Whether this is the high-resolution raster
        force_reload: Whether to force reload from disk even if cached
        
    Returns:
        lats: Array of latitude values
        lons: Array of longitude values
        population: 2D array of population values
    """
    global HIGH_RES_CACHE, LOW_RES_CACHE
    
    # Check if already cached
    if not force_reload:
        if is_high_res and HIGH_RES_CACHE is not None:
            logger.info("Using cached high-resolution raster data")
            return HIGH_RES_CACHE
        elif not is_high_res and LOW_RES_CACHE is not None:
            logger.info("Using cached low-resolution raster data")
            return LOW_RES_CACHE
    
    logger.info(f"Loading raster from: {raster_path}")
    
    try:
        with rasterio.open(raster_path) as src:
            height, width = src.shape
            logger.info(f"Raster dimensions: {width}x{height} pixels")
            
            # Read the data
            population = src.read(1)
            
            # Get bounds
            west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            logger.info(f"Raster bounds: {west}, {south}, {east}, {north}")
            
            # Create coordinate arrays
            lons = np.linspace(west, east, population.shape[1])
            lats = np.linspace(north, south, population.shape[0])  # Note: top to bottom for raster orientation
            
            # Replace nodata values with 0
            if src.nodata is not None:
                population[population == src.nodata] = 0
                
            # Convert negative values (if any) to 0
            population[population < 0] = 0
            
            logger.info(f"Loaded population raster with shape: {population.shape}")
            logger.debug(f"Latitude range: {min(lats)} to {max(lats)}")
            logger.debug(f"Longitude range: {min(lons)} to {max(lons)}")
            
            # Cache the results if appropriate
            result = (lats, lons, population)
            if is_high_res:
                HIGH_RES_CACHE = result
            else:
                LOW_RES_CACHE = result
                
            return result
            
    except Exception as e:
        logger.error(f"Error loading raster: {str(e)}", exc_info=True)
        raise

def create_iftar_polygon(east_lon, west_lon):
    """
    Create a polygon representing the Iftar zone
    
    Args:
        east_lon: Eastern edge longitude (earlier sunset)
        west_lon: Western edge longitude (later sunset)
        
    Returns:
        zone_lats: Latitude values for polygon corners
        zone_lons: Longitude values for polygon corners
    """
    zone_lats = []
    zone_lons = []
    
    # Check if we cross the date line (international date line)
    crosses_date_line = False
    if east_lon < west_lon:
        crosses_date_line = True
        logger.info("Iftar zone polygon crosses the international date line")
    
    # Calculate the longitude points for the polygon
    if crosses_date_line:
        # If we cross the date line, need to create two regions
        # First region: from west_lon to 180°
        lon_points1 = np.linspace(west_lon, 180, 50)
        # Second region: from -180° to east_lon
        lon_points2 = np.linspace(-180, east_lon, 50)
        lon_points = np.concatenate([lon_points1, lon_points2])
    else:
        # Normal case - create polygon between west_lon and east_lon
        lon_points = np.linspace(west_lon, east_lon, 100)
    
    # Top edge (at high latitude)
    for lon in lon_points:
        zone_lats.append(85)  # Using 85 degrees rather than 90 for better visualization
        zone_lons.append(lon)
    
    # Right edge (at east_lon)
    for lat in np.linspace(85, -85, 50):
        zone_lats.append(lat)
        zone_lons.append(east_lon)
    
    # Bottom edge (at low latitude)
    for lon in reversed(lon_points):
        zone_lats.append(-85)  # Using -85 degrees rather than -90 for better visualization
        zone_lons.append(lon)
    
    # Left edge (at west_lon)
    for lat in np.linspace(-85, 85, 50):
        zone_lats.append(lat)
        zone_lons.append(west_lon)
    
    return zone_lats, zone_lons

def compute_iftar_zone(current_time, window=10):
    """
    Compute iftar zone based on longitude band
    Returns points in the iftar zone and visualization data
    
    Args:
        current_time: Current time in minutes since midnight
        window: Width of the iftar window in minutes
    """
    global HIGH_RES_CACHE
    
    start_time = time.time()
    
    # Calculate the longitude band
    east_lon, west_lon = calculate_iftar_longitudes(current_time, window)
    center_lon = ((east_lon + west_lon) / 2) % 360
    if center_lon > 180:
        center_lon -= 360
        
    logger.info(f"Iftar band: east={east_lon:.2f}°, west={west_lon:.2f}°, center={center_lon:.2f}°")
    logger.info(f"Longitude window width: {abs(east_lon - west_lon):.2f}°")
    
    # Double-check our longitude band makes sense
    # For a 10-minute window, we expect a 2.5° band (0.25° per minute)
    expected_width = window * 0.25
    if abs(abs(east_lon - west_lon) - expected_width) > 1:
        logger.warning(f"Unexpected longitude band width. Expected {expected_width}° but got {abs(east_lon - west_lon):.2f}°")
    
    # Use cached high-resolution data instead of loading it again
    if HIGH_RES_CACHE is None:
        pop_lats, pop_lons, population = load_population_raster(HIGH_RES_POPULATION_RASTER, is_high_res=True)
    else:
        pop_lats, pop_lons, population = HIGH_RES_CACHE
    
    # Calculate indices that map from longitude values to array indices
    lon_resolution = pop_lons[1] - pop_lons[0]
    logger.info(f"Longitude resolution: {lon_resolution:.6f}° per pixel")
    
    # Create mask for the longitude band
    logger.info("Creating longitude mask...")
    
    # Check if the band crosses the date line (international date line)
    crosses_date_line = False
    if east_lon < west_lon:
        # This means we cross the date line
        # For example: east_lon = -179, west_lon = 179
        crosses_date_line = True
        logger.info("Iftar zone crosses the international date line")
    
    # Create the longitude mask using vectorized operations
    if crosses_date_line:
        # If we cross the date line, we want longitudes >= west_lon OR <= east_lon
        lon_mask = (pop_lons >= west_lon) | (pop_lons <= east_lon)
    else:
        # Normal case - we want longitudes between west_lon and east_lon
        lon_mask = (pop_lons >= west_lon) & (pop_lons <= east_lon)
    
    # Report on mask
    mask_sum = np.sum(lon_mask)
    logger.info(f"Created longitude mask with {mask_sum} columns in zone out of {len(pop_lons)} total")
    
    # Report actual band width based on mask
    if mask_sum > 0:
        band_width_pixels = mask_sum
        band_width_degrees = band_width_pixels * lon_resolution
        logger.info(f"Band width in pixels: {band_width_pixels}, which equals {band_width_degrees:.2f}°")
        
        # Check if the band width is reasonable
        if abs(band_width_degrees - expected_width) > 1:
            logger.warning(f"Band width in pixels ({band_width_degrees:.2f}°) doesn't match expected width ({expected_width}°)")
            
            # If the band is too wide (almost the whole globe), we have the comparison backwards
            if band_width_degrees > 350:
                logger.warning("Band is almost the entire globe. Inverting the mask.")
                lon_mask = ~lon_mask
                mask_sum = np.sum(lon_mask)
                band_width_pixels = mask_sum
                band_width_degrees = band_width_pixels * lon_resolution
                logger.info(f"After inversion: {mask_sum} columns, {band_width_degrees:.2f}°")
    
    # Handle special case where the zone is too small or not found
    if mask_sum < 5:
        logger.warning(f"Iftar zone is too small or not found: only {mask_sum} pixels")
        
    # OPTIMIZATION: Calculate the population in the zone using NumPy's optimized operations
    # This is much faster than looping through each row
    # For memory efficiency, still process in batches of rows
    total_population = 0
    batch_size = 1000  # Process 1000 rows at a time to avoid memory issues
    
    for i in range(0, population.shape[0], batch_size):
        end_idx = min(i + batch_size, population.shape[0])
        # Extract the batch of rows
        batch = population[i:end_idx, :]
        # Apply the mask to the batch and sum the result
        batch_sum = np.sum(batch[:, lon_mask])
        total_population += batch_sum
        
        # Log progress every 5000 rows or at the end
        if (i + batch_size >= 5000 and (i + batch_size) % 5000 == 0) or end_idx == population.shape[0]:
            logger.info(f"Processed {end_idx}/{population.shape[0]} rows, current sum: {total_population:,.0f}")
    
    logger.info(f"Total population in iftar zone: {total_population:,.0f}")
    
    # Create visualization elements
    # Create the band edges and center line for visualization (with higher density)
    band_edge_lats = []
    band_edge_lons = []
    center_line_lats = []
    center_line_lons = []
    
    # Add leading edge
    for lat in np.linspace(-85, 85, 100):
        band_edge_lats.append(lat)
        band_edge_lons.append(east_lon)
    
    # Add trailing edge
    for lat in np.linspace(-85, 85, 100):
        band_edge_lats.append(lat)
        band_edge_lons.append(west_lon)
    
    # Add center dotted line
    for lat in np.linspace(-85, 85, 100):
        center_line_lats.append(lat)
        center_line_lons.append(center_lon)
    
    # Create polygon for the iftar zone
    zone_lats, zone_lons = create_iftar_polygon(east_lon, west_lon)
    
    # Get the choropleth data from cache
    choropleth_data = load_choropleth_data()
    
    end_time = time.time()
    logger.info(f"Total computation time: {end_time - start_time:.2f} seconds")
    
    return {
        'total_population': total_population,
        'band_edge_lats': band_edge_lats,
        'band_edge_lons': band_edge_lons,
        'east_lon': east_lon,
        'west_lon': west_lon,
        'center_line_lats': center_line_lats,
        'center_line_lons': center_line_lons,
        'zone_lats': zone_lats,
        'zone_lons': zone_lons,
        'choropleth_data': choropleth_data
    }

def prepare_choropleth_data(low_res_data=None):
    """
    Pre-process the low-resolution raster for choropleth visualization
    
    Args:
        low_res_data: Pre-loaded low-resolution data (optional)
        
    Returns:
        A dictionary with choropleth data for visualization
    """
    global LOW_RES_CACHE
    
    if low_res_data is None and LOW_RES_CACHE is None:
        low_res_data = load_population_raster(LOW_RES_POPULATION_RASTER, is_high_res=False)
    elif low_res_data is None:
        low_res_data = LOW_RES_CACHE
    
    lats, lons, population = low_res_data
    
    # Enhanced visualization settings
    grid_lats = []
    grid_lons = []
    grid_values = []
    
    step_lat = max(1, len(lats) // 180)  # Aim for about 1 degree resolution
    step_lon = max(1, len(lons) // 360)
    
    logger.info(f"Preparing choropleth data with steps: lat={step_lat}, lon={step_lon}")
    
    # Create a mask for areas with very small population values to improve visualization
    population_threshold = np.percentile(population[population > 0], 10)  # Exclude bottom 10% of values
    
    for i in range(0, len(lats), step_lat):
        for j in range(0, len(lons), step_lon):
            if population[i, j] > population_threshold:
                grid_lats.append(lats[i])
                grid_lons.append(lons[j])
                grid_values.append(population[i, j])
    
    logger.info(f"Created choropleth data with {len(grid_lats)} points")
    
    return {
        'lats': grid_lats,
        'lons': grid_lons,
        'values': grid_values
    }

def load_choropleth_data():
    """
    Return the cached choropleth data
    
    Returns:
        A dictionary with choropleth data for visualization
    """
    global CHOROPLETH_DATA_CACHE
    
    # Try to use cache first
    if CHOROPLETH_DATA_CACHE is not None:
        logger.info("Using cached choropleth data")
        return CHOROPLETH_DATA_CACHE
    
    # If not cached, try to create it
    try:
        logger.info("Choropleth data not found in cache, creating now")
        CHOROPLETH_DATA_CACHE = prepare_choropleth_data()
        return CHOROPLETH_DATA_CACHE
    except Exception as e:
        logger.error(f"Error creating choropleth data: {str(e)}")
        # Return empty data if there's an error
        return {'lats': [], 'lons': [], 'values': []}

def convert_to_cartesian(lat, lon, radius=1.0):
    """
    Convert latitude and longitude to 3D cartesian coordinates
    for WebGL-accelerated rendering
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        radius: Radius of the globe (default: 1.0)
        
    Returns:
        x, y, z: Cartesian coordinates
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Convert to cartesian coordinates
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z

def pre_calculate_cartesian_coords(choropleth_data):
    """
    Pre-calculate cartesian coordinates for all points to avoid recalculation
    during rendering.
    
    Args:
        choropleth_data: Dictionary with lats, lons, and values
    
    Returns:
        Dictionary with x, y, z coordinates and original values
    """
    if not choropleth_data or not choropleth_data.get('lats'):
        return None
        
    x_points = []
    y_points = []
    z_points = []
    values = choropleth_data.get('values', [])
    
    # Radius slightly above the globe surface
    radius = 1.01
    
    # Convert each point
    for lat, lon in zip(choropleth_data['lats'], choropleth_data['lons']):
        x, y, z = convert_to_cartesian(lat, lon, radius)
        x_points.append(x)
        y_points.append(y)
        z_points.append(z)
    
    return {
        'x': x_points,
        'y': y_points,
        'z': z_points,
        'values': values
    }

def create_webgl_globe(data, center_lon=0, center_lat=25, current_time=0, zoom_level=1.2, use_webgl=True):
    """
    Create a WebGL-accelerated 3D globe visualization
    
    Args:
        data: Dictionary containing visualization data
        center_lon: Center longitude for initial view
        center_lat: Center latitude for initial view
        current_time: Current time in minutes (for uirevision)
        zoom_level: Camera distance multiplier (smaller = closer zoom)
        use_webgl: Whether to use WebGL acceleration
        
    Returns:
        fig: Plotly figure with 3D globe
    """
    global CARTESIAN_COORDS_CACHE
    
    # Define POP_THRESHOLD at the top level of the function to ensure it's available
    POP_THRESHOLD = 25000
    
    logger.info("Creating WebGL-accelerated 3D globe")
    
    # Extract data
    choropleth_data = data.get('choropleth_data', {})
    zone_lats = data.get('zone_lats', [])
    zone_lons = data.get('zone_lons', [])
    band_edge_lats = data.get('band_edge_lats', [])
    band_edge_lons = data.get('band_edge_lons', [])
    center_line_lats = data.get('center_line_lats', [])
    center_line_lons = data.get('center_line_lons', [])
    east_lon = data.get('east_lon', 0)
    west_lon = data.get('west_lon', 0)
    
    # Create a blank 3D figure for the globe
    fig = go.Figure()
    
    # Create background sphere with cosmic appearance
    if use_webgl:
        # Create a larger background sphere 
        r_bg = 2.0  # Reduced radius to 2.0 to zoom in more
        
        # Background sphere coordinates (low resolution is fine for background)
        phi_bg = np.linspace(0, 2*np.pi, 30)
        theta_bg = np.linspace(0, np.pi, 20)
        phi_bg, theta_bg = np.meshgrid(phi_bg, theta_bg)
        
        x_bg = r_bg * np.sin(theta_bg) * np.cos(phi_bg)
        y_bg = r_bg * np.sin(theta_bg) * np.sin(phi_bg)
        z_bg = r_bg * np.cos(theta_bg)
        
        # Create a gradient with higher opacity
        # Use a darker color palette that contrasts better with data points
        colorscale_bg = [
            [0, 'rgb(5, 5, 20)'],        # Very dark blue
            [0.5, 'rgb(10, 10, 30)'],    # Dark blue-purple
            [1, 'rgb(15, 10, 25)']       # Dark purple
        ]
        
        # Add background sphere with higher opacity to make globe visible
        fig.add_trace(go.Surface(
            x=x_bg, y=y_bg, z=z_bg,
            colorscale=colorscale_bg,
            opacity=0.8,  # Slightly increased opacity
            showscale=False,
            hoverinfo='none',
            lighting=dict(
                ambient=0.3,  # Increased ambient light
                diffuse=0.0,  # No diffuse light to avoid washing out colors
                fresnel=0.2,  # Slight edge highlighting
                specular=0.1,  # Very slight specular highlights
                roughness=0.9  # Rough surface (less reflective)
            )
        ))
    
    # Create base globe mesh coordinates
    u = np.linspace(0, 2*np.pi, 50)  # Reduced from 100
    v = np.linspace(0, np.pi, 30)    # Reduced from 50

    x_globe = 1 * np.outer(np.cos(u), np.sin(v))
    y_globe = 1 * np.outer(np.sin(u), np.sin(v))
    z_globe = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create base globe
    if use_webgl:
        # Add primary globe sphere
        fig.add_trace(go.Surface(
            x=x_globe, y=y_globe, z=z_globe,
            colorscale=[[0, 'rgb(60, 60, 80)'], [1, 'rgb(60, 60, 80)']],  # Lighter gray-blue for better contrast
            opacity=0.9,  # Increased opacity
            showscale=False,
            hoverinfo='none',
            lighting=dict(
                ambient=0.7,  # Increased ambient to brighten the globe
                diffuse=0.5,  # More diffuse lighting
                fresnel=0.2,
                specular=0.5,  # More specular to make globe pop
                roughness=0.5
            )
        ))
    
    # Use pre-calculated cartesian coordinates if available
    if CARTESIAN_COORDS_CACHE and CARTESIAN_COORDS_CACHE.get('x'):
        # OPTIMIZATION: More efficient filtering of visible points
        # Instead of calculating dot products for every point, use octree-like approach
        # by filtering in chunks based on longitude/latitude first
        
        # Quick pre-filter based on longitude and latitude difference from view center
        if len(CARTESIAN_COORDS_CACHE['x']) > 1000:  # Only apply filtering to large datasets
            visible_indices = []
            
            # Convert center_lat, center_lon to radians for faster calculations
            center_lat_rad = np.radians(center_lat)
            center_lon_rad = np.radians(center_lon)
            
            # Make sure we have valid lat/lon data in the cache
            if ('lats' not in CARTESIAN_COORDS_CACHE or 
                'lons' not in CARTESIAN_COORDS_CACHE or 
                'values' not in CARTESIAN_COORDS_CACHE or
                len(CARTESIAN_COORDS_CACHE['lats']) == 0):
                logger.error("Missing or empty lat/lon data in cartesian coordinates cache")
                # We'll create a simple globe without points in this case
                x_points = []
                y_points = []
                z_points = []
                original_lats = []
                original_lons = []
                capped_values = []
                hover_text = []
            else:
                # Normalize original coordinates for efficient operations
                original_lats = np.array(CARTESIAN_COORDS_CACHE.get('lats', []))
                original_lons = np.array(CARTESIAN_COORDS_CACHE.get('lons', []))
                capped_values = np.array(CARTESIAN_COORDS_CACHE.get('values', []))
                
                # Use dot product to filter points on the visible hemisphere
                # (much faster than comparing distances)
                x_points = np.array(CARTESIAN_COORDS_CACHE.get('x', []))
                y_points = np.array(CARTESIAN_COORDS_CACHE.get('y', []))
                z_points = np.array(CARTESIAN_COORDS_CACHE.get('z', []))
                
                # Calculate dot product between each point and the camera vector
                # The camera is looking at the center of the globe from (center_lon, center_lat)
                cam_x = np.cos(center_lon_rad) * np.cos(center_lat_rad)
                cam_y = np.sin(center_lon_rad) * np.cos(center_lat_rad)
                cam_z = np.sin(center_lat_rad)
                
                # Calculate dot product efficiently
                dot_products = x_points * cam_x + y_points * cam_y + z_points * cam_z
                
                # Points on the visible hemisphere have dot product ≥ 0
                visible_mask = dot_products >= -0.2  # Include some points just over the edge
                
                # Extract visible points - only if we have data
                if len(x_points) > 0 and len(visible_mask) == len(x_points):
                    x_points = x_points[visible_mask]
                    y_points = y_points[visible_mask]
                    z_points = z_points[visible_mask]
                    capped_values = capped_values[visible_mask]
                    
                    # Only apply mask if arrays have the same length
                    if len(original_lats) == len(visible_mask):
                        original_lats = original_lats[visible_mask]
                        original_lons = original_lons[visible_mask]
                    else:
                        logger.warning(f"Size mismatch: original_lats={len(original_lats)}, mask={len(visible_mask)}")
                        # In this case, reset to empty arrays to avoid errors
                        original_lats = np.array([])
                        original_lons = np.array([])
                
                # Generate hover text only if we have data
                hover_text = []
                if len(x_points) > 0 and len(original_lats) == len(x_points):
                    for i in range(len(x_points)):
                        pop = int(capped_values[i])
                        lat = original_lats[i]
                        lon = original_lons[i]
                        hover_text.append(f"Muslim Population: {pop:,}<br>Lat: {lat:.2f}°, Lon: {lon:.2f}°")
                
                logger.info(f"Efficient filtering: Showing {len(x_points)} of {len(CARTESIAN_COORDS_CACHE['x'])} points")
            
            # Check which points are in the Iftar zone
            if east_lon is not None and west_lon is not None:
                # Check if zone crosses the antimeridian (International Date Line)
                if east_lon < west_lon:
                    # Zone crosses the antimeridian
                    in_iftar_zone = np.logical_or(original_lons >= west_lon, original_lons <= east_lon)
                else:
                    in_iftar_zone = np.logical_and(original_lons >= west_lon, original_lons <= east_lon)
            
            # Apply pop-out effect - OPTIMIZATION: Vectorized operations
            pop_factors = 1.0 + (0.2 * in_iftar_zone)  # 20% pop-out
            
            # Apply pop factors to coordinates
            z_with_pop = z_points * pop_factors
            
            # Use logarithmic scale for marker sizes (smaller in 3D) - OPTIMIZATION: Vectorized
            marker_base_size = 2.0
            marker_max_addon = 4.0
            log_values = np.log1p(capped_values)
            log_max = np.log1p(POP_THRESHOLD)
            size_factors = log_values / log_max
            marker_sizes = marker_base_size + (marker_max_addon * size_factors)
            
            # Apply size adjustments for Iftar zone points - OPTIMIZATION: Vectorized
            marker_sizes = marker_sizes * pop_factors
            
            # Create color array for neon glow effect for points in the iftar zone
            colors_rgb = np.zeros((len(capped_values), 3))
            # Map values to the colorscale (blue to red gradient)
            norm_values = capped_values / POP_THRESHOLD
            # Blue (for low values) to red (for high values)
            colors_rgb[:, 0] = np.minimum(1.0, norm_values * 2.0)  # Red channel
            colors_rgb[:, 2] = np.maximum(0.0, 1.0 - norm_values * 2.0)  # Blue channel
            colors_rgb[:, 1] = np.minimum(norm_values * 2.0, 2.0 - norm_values * 2.0)  # Green channel
            
            # Brighten colors for points in the iftar zone for a neon glow effect
            for i in range(3):  # For each RGB channel
                colors_rgb[:, i] = np.where(
                    in_iftar_zone,
                    np.minimum(1.0, colors_rgb[:, i] * 1.5),  # Increase brightness by 50% for neon effect
                    colors_rgb[:, i]
                )
            
            # OPTIMIZATION: Only show a maximum number of points for performance
            MAX_POINTS_TO_RENDER = 5000
            if len(x_points) > MAX_POINTS_TO_RENDER:
                # If we have too many points, sample them
                indices = np.random.choice(len(x_points), MAX_POINTS_TO_RENDER, replace=False)
                x_points = x_points[indices]
                y_points = y_points[indices]
                z_with_pop = z_with_pop[indices]
                capped_values = capped_values[indices]
                hover_text = [hover_text[i] for i in indices]
                marker_sizes = marker_sizes[indices]
                colors_rgb = colors_rgb[indices]
                logger.info(f"Limited to {MAX_POINTS_TO_RENDER} points for performance")
            
            # Create RGB strings for each point
            color_strings = [f'rgb({r*255:.0f}, {g*255:.0f}, {b*255:.0f})' for r, g, b in colors_rgb]
            
            # Create a more elegant colorscale (keep the definition but use custom colors)
            elegant_colorscale = [
                [0.0, "rgba(31, 58, 147, 0.7)"],     # Deep blue
                [0.2, "rgba(58, 76, 192, 0.75)"],    # Royal blue
                [0.4, "rgba(111, 107, 213, 0.8)"],   # Purple-blue
                [0.6, "rgba(188, 98, 173, 0.85)"],   # Purple-pink
                [0.8, "rgba(239, 94, 124, 0.9)"],    # Coral
                [1.0, "rgba(249, 142, 82, 0.95)"]    # Orange
            ]
            
            # OPTIMIZATION: Use more WebGL-friendly settings
            fig.add_trace(go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_with_pop,
                mode='markers',
                marker=dict(
                    size=marker_sizes,
                    color=color_strings,
                    opacity=0.85,  # Use single opacity value instead of array
                    colorbar=dict(
                        title=dict(
                            text="Muslim Population",
                            side="right",
                            font=dict(size=12)
                        ),
                        x=1.02,
                        xpad=0,
                        y=0.3,
                        tickfont=dict(size=10),
                        ticks="outside",
                        len=0.6,
                        tickvals=[0, POP_THRESHOLD/5, POP_THRESHOLD*2/5, POP_THRESHOLD*3/5, POP_THRESHOLD*4/5, POP_THRESHOLD],
                        ticktext=["0", "5k", "10k", "15k", "20k", "≥25k"]
                    ),
                    cmin=0,
                    cmax=POP_THRESHOLD,
                ),
                name='Global Muslim Population',
                hoverinfo='text',
                hovertext=hover_text,
                # OPTIMIZATION: WebGL-specific performance settings
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=10,
                    font_family="Arial"
                )
            ))
    
    # Optimize zone rendering by reducing the number of points
    if zone_lats and zone_lons:
        # OPTIMIZATION: Sample points for better performance
        sample_rate = max(1, len(zone_lats) // 100)  # Aim for about 100 points
        
        x_zone = []
        y_zone = []
        z_zone = []
        
        # Use slightly larger radius for the zone to be visible above the globe
        radius = 1.015
        
        for lat, lon in zip(zone_lats[::sample_rate], zone_lons[::sample_rate]):
            x, y, z = convert_to_cartesian(lat, lon, radius)
            x_zone.append(x)
            y_zone.append(y)
            z_zone.append(z)
        
        # Add Iftar zone outline
        fig.add_trace(go.Scatter3d(
            x=x_zone,
            y=y_zone,
            z=z_zone,
            mode='lines',
            line=dict(color='rgb(25, 230, 25)', width=5),
            name='Iftar Zone Boundary',
            hoverinfo='none'
        ))
    
    # Add band edges to show the exact iftar longitude limits
    if band_edge_lats and band_edge_lons:
        # East band edge
        east_x = []
        east_y = []
        east_z = []
        
        # Use an even larger radius for the east band edge
        radius_edge = 1.02
        
        for lat, lon in zip(band_edge_lats, band_edge_lons):
            # Only plot the eastern edge
            if abs(lon - east_lon) < 0.001:
                x, y, z = convert_to_cartesian(lat, lon, radius_edge)
                east_x.append(x)
                east_y.append(y)
                east_z.append(z)
        
        # Add east band edge
        fig.add_trace(go.Scatter3d(
            x=east_x,
            y=east_y,
            z=east_z,
            mode='lines',
            line=dict(color='rgb(255, 50, 50)', width=4),
            name='Eastern Edge (Sunset)',
            hoverinfo='none'
        ))
        
        # West band edge
        west_x = []
        west_y = []
        west_z = []
        
        for lat, lon in zip(band_edge_lats, band_edge_lons):
            # Only plot the western edge
            if abs(lon - west_lon) < 0.001:
                x, y, z = convert_to_cartesian(lat, lon, radius_edge)
                west_x.append(x)
                west_y.append(y)
                west_z.append(z)
        
        # Add west band edge
        fig.add_trace(go.Scatter3d(
            x=west_x,
            y=west_y,
            z=west_z,
            mode='lines',
            line=dict(color='rgb(255, 165, 0)', width=4),
            name='Western Edge (Maghrib)',
            hoverinfo='none'
        ))
    
    # Add center line to help visualize the center of the iftar zone
    if center_line_lats and center_line_lons:
        center_x = []
        center_y = []
        center_z = []
        
        # Use an even larger radius for the center line for visibility
        radius_center = 1.01
        
        for lat, lon in zip(center_line_lats, center_line_lons):
            x, y, z = convert_to_cartesian(lat, lon, radius_center)
            center_x.append(x)
            center_y.append(y)
            center_z.append(z)
        
        # Add center line
        fig.add_trace(go.Scatter3d(
            x=center_x,
            y=center_y,
            z=center_z,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash'),
            name='Zone Center',
            hoverinfo='none'
        ))
        
    logger.info(f"Rendering {len(x_points) if 'x_points' in locals() else 0} points in 3D")

    # Update the scene layout for WebGL version
    fig.update_scenes(
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        aspectmode='data',
        dragmode='orbit',
        # Set camera to zoom in on the globe with user-defined zoom level
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(
                # Use zoom_level parameter for camera distance
                x=zoom_level * math.sin(math.radians(center_lon)) * math.cos(math.radians(center_lat)),
                y=zoom_level * -math.cos(math.radians(center_lon)) * math.cos(math.radians(center_lat)),
                z=zoom_level * math.sin(math.radians(center_lat))
            ),
            projection=dict(type="perspective")
        ),
        # Force scene update during animation while preserving camera position
        uirevision=f'time-{current_time}-{center_lon}-{center_lat}'
    )

    # Add a proper background color to the scene instead of using lighting
    fig.update_layout(
        scene=dict(
            bgcolor='rgb(5, 5, 20)'  # Very dark blue matching our background sphere
        )
    )
    
    logger.info("Created WebGL-accelerated 3D globe with optimized performance")
    return fig

# Create Dash app
app = Dash(__name__, suppress_callback_exceptions=True)
app.config.suppress_callback_exceptions = True

# Define app layout with a beautiful theme
app.layout = html.Div([
    # Add Font Awesome
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    ),
    
    html.Div([
        html.H1("Global Iftar Zone Visualization", 
                style={'textAlign': 'center', 'color': '#1E4620', 'fontFamily': 'Arial'}),
        html.P("Visualizing Muslim populations breaking their fast during Ramadan across the globe",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': '1.2em'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
    
    # Add toggle for WebGL 3D mode
    html.Div([
        html.Label("Visualization Mode:", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#333', 'marginRight': '10px'}),
        dcc.RadioItems(
            id='rendering-mode',
            options=[
                {'label': ' Standard Globe', 'value': 'standard'},
                {'label': ' WebGL 3D (Experimental)', 'value': 'webgl'}
            ],
            value='standard',
            inline=True,
            style={'fontSize': '1em', 'color': '#333'}
        )
    ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px', 'display': 'flex', 'alignItems': 'center'}),
    
    html.Div([
        dcc.Graph(id="globe-visualization", style={'height': '70vh'}),
        
        # Animation control panel (positioned at bottom left of the map)
        html.Div([
            # Animation button
            html.Button(
                id='animation-button',
                children=[html.I(className="fa fa-play"), " Animate"],
                style={
                    'backgroundColor': '#1E4620', 
                    'color': 'white', 
                    'border': 'none',
                    'padding': '10px 15px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'gap': '5px',
                    'fontWeight': 'bold',
                    'marginBottom': '10px'
                }
            ),
            # Add zoom level slider
            html.Div([
                html.Label("Zoom Level:", style={'fontSize': '0.9em', 'color': 'white', 'marginBottom': '5px'}),
                dcc.Slider(
                    id="zoom-slider",
                    min=0.5,
                    max=3.0,
                    step=0.1,
                    value=1.2,
                    marks={
                        0.5: {'label': 'Far', 'style': {'color': 'white'}},
                        1.0: {'label': '1.0', 'style': {'color': 'white'}},
                        2.0: {'label': '2.0', 'style': {'color': 'white'}},
                        3.0: {'label': 'Close', 'style': {'color': 'white'}}
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={
                'backgroundColor': 'rgba(30, 70, 32, 0.8)',
                'padding': '10px',
                'borderRadius': '5px',
                'width': '200px'
            }),
            
            # Add time increment slider
            html.Div([
                html.Label("Time Increment (minutes):", style={'fontSize': '0.9em', 'color': 'white', 'marginBottom': '5px', 'marginTop': '10px'}),
                html.Div([
                    dcc.Slider(
                        id="increment-slider",
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        marks={
                            1: {'label': '1 min', 'style': {'color': 'white'}},
                            5: {'label': '5 min', 'style': {'color': 'white'}},
                            10: {'label': '10 min', 'style': {'color': 'white'}}
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '100%', 'padding': '10px 0'}),
            ], style={'width': '100%', 'padding': '10px 0'}),
        ], style={
            'position': 'absolute',
            'bottom': '30px',
            'left': '40px',
            'zIndex': '1000',
            'display': 'flex',
            'flexDirection': 'column'
        }),
        
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 
              'borderRadius': '10px', 'margin': '10px', 'position': 'relative'}),
    
    html.Div([
        html.Div([
            html.Label("Current Time:", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#333'}),
            html.Div(id="time-display", style={'fontSize': '1.3em', 'color': '#1E4620'}),
            dcc.Slider(
                id="time-slider",
                min=0,
                max=1440,
                step=1,  # Changed from 10 to 1 minute steps
                value=1150,  # Default: 19:10
                marks={i: f"{i//60}:{i%60:02d}" for i in range(0, 1441, 120)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Iftar Window (minutes):", style={'fontSize': '1.1em', 'fontWeight': 'bold', 'color': '#333'}),
            dcc.Slider(
                id="window-slider",
                min=1,
                max=30,
                step=1,
                value=10,
                marks={i: str(i) for i in range(0, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 
              'borderRadius': '10px', 'margin': '10px', 'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.Div([
            html.H3("Controls", style={'color': '#1E4620'}),
            html.Button('Center on Iftar Zone', id='center-button', 
                       style={'backgroundColor': '#1E4620', 'color': 'white', 'border': 'none', 
                              'padding': '10px', 'borderRadius': '5px', 'margin': '10px 0'}),
        ], style={'width': '100%', 'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 
              'borderRadius': '10px', 'margin': '10px'}),
    
    html.Div([
        html.P("This visualization shows the regions where Muslims are breaking their fast during Ramadan. "
              "The highlighted band represents the current Iftar zone based on longitude.",
              style={'fontSize': '1em', 'color': '#666'}),
        html.P("The model uses a simplified approach where the Iftar zone is represented as a band of longitudes moving across the Earth.",
              style={'fontSize': '0.9em', 'color': '#999', 'fontStyle': 'italic'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
    
    # Keep statistics in hidden divs for callback dependencies
    html.Div(id="total-population", style={'display': 'none'}),
    html.Div(id="longitude-range", style={'display': 'none'}),
    
    # Animation interval component (hidden)
    dcc.Interval(
        id='animation-interval',
        interval=1250,  # Increased from 1000 to 1250 milliseconds for smoother animation
        n_intervals=0,
        disabled=True
    ),
    
    # Store animation state
    dcc.Store(id='animation-running', data=False),
    dcc.Store(id='time-increment', data=1),  # Store for time increment value
    
    # Loading component for visual feedback
    dcc.Loading(
        id="loading-indicator",
        type="circle",
        children=[html.Div(id="loading-output")]
    )
])

# Update time display
@app.callback(
    Output("time-display", "children"),
    Input("time-slider", "value")
)
def update_time_display(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d} UTC"

# Main callback to update the globe
@app.callback(
    [Output("globe-visualization", "figure"),
     Output("total-population", "children"),
     Output("longitude-range", "children"),
     Output("loading-output", "children", allow_duplicate=True)],
    [Input("time-slider", "value"),
     Input("window-slider", "value"),
     Input("center-button", "n_clicks"),
     Input("animation-interval", "n_intervals"),
     Input("rendering-mode", "value"),
     Input("zoom-slider", "value")],  # Add zoom slider input
    [State("globe-visualization", "figure"),
     State("animation-running", "data")],
    prevent_initial_call='initial_duplicate'
)
def update_globe(current_time, window, n_clicks, n_intervals, rendering_mode, zoom_level, current_fig, is_animating):
    logger.info(f"Updating visualization with time={current_time}, window={window}, mode={rendering_mode}, zoom={zoom_level}")
    
    # OPTIMIZATION: For animation frames, use faster update path
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
    is_animation_update = trigger_id == 'animation-interval.n_intervals' and is_animating
    is_zoom_update = trigger_id == 'zoom-slider.value'
    
    # Convert current_time to hours:minutes format for display
    hours = current_time // 60
    minutes = current_time % 60
    time_str = f"{hours:02d}:{minutes:02d}"
    
    # Important: For animation updates, we need to always recalculate the iftar zone
    # as this is what creates the sliding effect
    try:
        logger.info(f"Computing iftar zone for time={current_time}, window={window}")
        iftar_data = compute_iftar_zone(current_time=current_time, window=window)
        
        # We only want to cache cartesian coordinates calculation, not the iftar zone boundaries
        if not is_animation_update and hasattr(update_globe, 'last_cartesian_coords'):
            # Reuse the precomputed cartesian coordinates but update the iftar zone information
            iftar_data['choropleth_data']['cartesian_coords'] = update_globe.last_cartesian_coords
        
        # Cache the cartesian coordinate data for future frames (not the entire iftar_data)
        if 'cartesian_coords' in iftar_data['choropleth_data']:
            update_globe.last_cartesian_coords = iftar_data['choropleth_data']['cartesian_coords']
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        # Create a simple error figure regardless of rendering mode
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='text',
            text=[f'Error processing data: {str(e)}'],
            textposition='middle center',
            textfont=dict(size=16, color='red')
        ))
        fig.update_layout(
            title=dict(text="Error: Data processing failed"),
            height=700,
            margin=dict(r=10, t=50, l=10, b=10),
        )
        return fig, f"Error: {str(e)}", "", ""
    
    # Extract data from the result
    total_population = iftar_data['total_population']
    band_edge_lats = iftar_data['band_edge_lats']
    band_edge_lons = iftar_data['band_edge_lons']
    east_lon = iftar_data['east_lon']
    west_lon = iftar_data['west_lon']
    center_line_lats = iftar_data['center_line_lats']
    center_line_lons = iftar_data['center_line_lons']
    zone_lats = iftar_data['zone_lats']
    zone_lons = iftar_data['zone_lons']
    choropleth_data = iftar_data['choropleth_data']
    
    # Calculate the center point for the globe view (middle of iftar zone)
    center_lon = (east_lon + west_lon) / 2
    if abs(east_lon - west_lon) > 180:  # Handle case where zone crosses the date line
        center_lon = (center_lon + 180) % 360 - 180
    center_lat = 25  # Center around populated areas
    
    # Choose rendering mode based on user selection
    if rendering_mode == 'webgl':
        try:
            # OPTIMIZATION: For animation frames, preserve camera position by reusing figure
            preserve_viewstate = (is_animation_update or is_zoom_update) and current_fig is not None
            
            # Use WebGL-accelerated 3D globe with user-defined zoom level
            fig = create_webgl_globe(
                data=iftar_data, 
                center_lon=center_lon, 
                center_lat=center_lat,
                current_time=current_time,
                zoom_level=zoom_level,  # Pass zoom level parameter
                use_webgl=True
            )
            
            # If preserving viewstate, use the camera position from the current figure
            if preserve_viewstate and current_fig and 'layout' in current_fig and 'scene' in current_fig['layout'] and 'camera' in current_fig['layout']['scene']:
                # Only preserve camera position if not a zoom update
                if not is_zoom_update:
                    # Create a deep copy of the camera settings to prevent reference issues
                    camera_settings = dict(current_fig['layout']['scene']['camera'])
                    
                    # Update just the camera position while keeping everything else fresh
                    fig.update_layout(
                        scene=dict(
                            camera=camera_settings,
                        ),
                        # Force an update to ensure redraw happens
                        uirevision=f"camera-{n_intervals}" if is_animation_update else "constant"
                    )
                    logger.info(f"Preserved camera position for animation frame at time {time_str}")
            
            # Add title to the layout
            fig.update_layout(
                title=dict(
                    text=f"Iftar Zone at {time_str} UTC (Window: {window} min)",
                    x=0.5,
                    font=dict(
                        family="Arial",
                        size=24,
                        color="#1E4620"
                    )
                ),
                height=700,
            )
            
            # Add statistics annotation to WebGL view
            stats_text = (
                f"<b>Statistics:</b><br>" +
                f"Muslim population breaking fast: <b>{int(total_population):,}</b><br>" +
                f"Longitude band: <b>{east_lon:.1f}° to {west_lon:.1f}°</b> ({window} minutes)"
            )
            
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(size=16, color="#1E4620", family="Arial", weight="bold"),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#1E4620",
                borderwidth=2,
                borderpad=10,
                width=380,
                height=100
            )
            
            logger.info("Using WebGL 3D globe rendering")
            
        except Exception as e:
            logger.error(f"Error creating WebGL globe: {str(e)}", exc_info=True)
            # Fall back to standard view if WebGL fails
            rendering_mode = 'standard'
            logger.warning("Falling back to standard globe rendering")

    # Use standard rendering if not WebGL or if WebGL failed
    if rendering_mode == 'standard':
        # Create an empty figure with globe projection
        fig = go.Figure()
        
        # Set up the globe projection
        fig.update_geos(
            projection_type="orthographic",
            projection_rotation=dict(
                lon=center_lon,
                lat=center_lat,
                roll=0
            ),
            showland=True,
            landcolor="rgba(230, 230, 220, 1)",
            showocean=True,
            oceancolor="rgba(210, 230, 240, 1)",
            showcountries=True,
            countrycolor="rgba(120, 120, 120, 0.5)",
            showcoastlines=True,
            coastlinecolor="rgba(80, 80, 80, 0.5)",
            showframe=False,
            bgcolor='rgba(250, 250, 250, 0.9)'
        )
        
        # Set up basic layout
        fig.update_layout(
            title=dict(
                text=f"Iftar Zone at {time_str} UTC (Window: {window} min)",
                x=0.5,
                font=dict(
                    family="Arial",
                    size=24,
                    color="#1E4620"
                )
            ),
            height=700,
            margin=dict(r=10, t=50, l=10, b=10),
            paper_bgcolor='rgba(250, 250, 250, 0.9)',
            plot_bgcolor='rgba(250, 250, 250, 0.9)',
            dragmode='orbit'  # Enable smooth globe dragging
        )
        
        # Add statistics annotation to top left corner of map
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            text="Statistics loading...",
            showarrow=False,
            font=dict(size=16, color="#1E4620", family="Arial", weight="bold"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#1E4620",
            borderwidth=2,
            borderpad=10,  # Increased padding from 8 to 10
            width=380,     # Increased width from 320 to 380
            height=100     # Increased height from 80 to 100
        )
        
        # Implement level-of-detail rendering for better performance
        # Get the current view bounds
        view_bounds = {
            'lon': center_lon,
            'lat': center_lat,
            'visible_range': 180  # In orthographic view, only half the globe is visible
        }
        
        # Add choropleth data (global Muslim population) as a heatmap
        if choropleth_data['lats'] and choropleth_data['lons'] and choropleth_data['values']:
            # PERFORMANCE OPTIMIZATION: Filter points to only those potentially visible
            visible_points = []
            visible_lats = []
            visible_lons = []
            visible_values = []
            
            # Determine which points are potentially visible
            for i, (lat, lon, val) in enumerate(zip(
                choropleth_data['lats'], 
                choropleth_data['lons'], 
                choropleth_data['values']
            )):
                # Points are potentially visible if they're within 90 degrees of the center
                # This is a simplification for the orthographic projection
                dist = abs(lon - center_lon)
                if dist > 180:
                    dist = 360 - dist
                    
                if dist <= 90 and abs(lat - center_lat) <= 90:
                    visible_lats.append(lat)
                    visible_lons.append(lon)
                    visible_values.append(val)
            
            logger.info(f"Rendering {len(visible_lats)} points out of {len(choropleth_data['lats'])} total points")
            
            # Scale marker sizes based on population
            if visible_values:
                # Cap population values at 25000 for better visualization
                POP_THRESHOLD = 25000
                capped_values = np.array(visible_values)
                # Create a copy of the original values for the hover text
                original_values = np.copy(capped_values)
                
                # Cap the values for color scale but keep original for hover text
                capped_values = np.minimum(capped_values, POP_THRESHOLD)
                
                # Define custom color scale
                # This gives more distinction in the lower ranges while keeping high population areas consistent
                max_value = POP_THRESHOLD
                min_value = np.min(capped_values[capped_values > 0]) if len(capped_values[capped_values > 0]) > 0 else 0
                
                # Use logarithmic scale for marker sizes to better represent the range
                # Scale from 3 to 12 pixels
                marker_sizes = 3 + 9 * (np.log1p(capped_values) / np.log1p(POP_THRESHOLD))
                
                # Create hover text with formatted population values (original, not capped)
                hover_text = []
                for val in original_values:
                    if val >= POP_THRESHOLD:
                        hover_text.append(f"Muslim population: {int(val):,} (≥5,000)")
                    else:
                        hover_text.append(f"Muslim population: {int(val):,}")
                
                # Create a more elegant colorscale (using a sophisticated gradient)
                elegant_colorscale = [
                    [0.0, "rgba(31, 58, 147, 0.7)"],     # Deep blue
                    [0.2, "rgba(58, 76, 192, 0.75)"],    # Royal blue
                    [0.4, "rgba(111, 107, 213, 0.8)"],   # Purple-blue
                    [0.6, "rgba(188, 98, 173, 0.85)"],   # Purple-pink
                    [0.8, "rgba(239, 94, 124, 0.9)"],    # Coral
                    [1.0, "rgba(249, 142, 82, 0.95)"]    # Orange
                ]
                
                try:
                    # For WebGL rendering - note: Plotly doesn't directly support Scattergl with geographical coords
                    # so we use Scattergeo with a faster renderer
                    fig.add_trace(go.Scattergeo(
                        lon=visible_lons,
                        lat=visible_lats,
                        mode='markers',
                        marker=dict(
                            size=marker_sizes,
                            color=capped_values,
                            colorscale=elegant_colorscale,
                            opacity=0.8,
                            showscale=True,
                            colorbar=dict(
                                title=dict(
                                    text="Muslim Population",
                                    side="right",
                                    font=dict(size=12)
                                ),
                                x=1.02,  # Moved further to the right
                                xpad=0,  # Reduced padding
                                y=0.3,   # Moved down to avoid toolbar overlap
                                tickfont=dict(size=10),
                                ticks="outside",
                                len=0.6,  # Slightly smaller
                                tickvals=[min_value, POP_THRESHOLD/5, POP_THRESHOLD*2/5, POP_THRESHOLD*3/5, POP_THRESHOLD*4/5, POP_THRESHOLD],
                                ticktext=["0", "5k", "10k", "15k", "20k", "≥25k"]
                            ),
                            cmin=min_value,
                            cmax=POP_THRESHOLD,
                        ),
                        name='Global Muslim Population',
                        hoverinfo='text',
                        hovertext=hover_text
                    ))
                    
                    # Apply "pop-out" effect for points in Iftar zone
                    # First identify which points are in the Iftar zone
                    in_iftar_zone = np.zeros_like(original_lons, dtype=bool)
                    
                    # Check which points are in the Iftar zone
                    if east_lon is not None and west_lon is not None and len(original_lons) > 0:
                        try:
                            # Make sure original_lons is a NumPy array to support comparison with floats
                            if isinstance(original_lons, list):
                                original_lons = np.array(original_lons)
                            
                            # Check if zone crosses the antimeridian (International Date Line)
                            if east_lon < west_lon:
                                # Zone crosses the antimeridian
                                in_iftar_zone = np.logical_or(original_lons >= west_lon, original_lons <= east_lon)
                            else:
                                in_iftar_zone = np.logical_and(original_lons >= west_lon, original_lons <= east_lon)
                        except TypeError as e:
                            logger.error(f"Type error when comparing longitudes: {e}")
                            # Create an array of False values as fallback
                            in_iftar_zone = np.zeros_like(original_lons, dtype=bool)
                    
                    # Create a popped-out visualization of points in the Iftar zone
                    iftar_lats = [lat for lat, is_in_zone in zip(visible_lats, in_iftar_zone) if is_in_zone]
                    iftar_lons = [lon for lon, is_in_zone in zip(visible_lons, in_iftar_zone) if is_in_zone]
                    iftar_values = [val for val, is_in_zone in zip(visible_values, in_iftar_zone) if is_in_zone]
                    
                    if iftar_lats:  # If any points are in the Iftar zone
                        # Cap values for consistency
                        iftar_capped = np.minimum(np.array(iftar_values), POP_THRESHOLD)
                        iftar_sizes = 4 + 10 * (np.log1p(iftar_capped) / np.log1p(POP_THRESHOLD))  # Slightly larger
                        
                        # Add highlighted points in Iftar zone
                        fig.add_trace(go.Scattergeo(
                            lon=iftar_lons,
                            lat=iftar_lats,
                            mode='markers',
                            marker=dict(
                                size=iftar_sizes,
                                color=iftar_capped,
                                colorscale=elegant_colorscale,
                                opacity=1.0,  # Full opacity
                                showscale=False,  # Don't show second colorbar
                            ),
                            name='Breaking Fast Now',
                            hoverinfo='text',
                            hovertext=[f"Muslim population breaking fast: {int(val):,}" for val in iftar_values]
                        ))
                    
                    # Add WebGL support through layout settings instead
                    fig.update_layout(
                        geo=dict(
                            # Enable fast rendering with higher performance settings
                            resolution=50,  # Lower resolution for better performance
                            showframe=False,
                            showcoastlines=True,
                            coastlinecolor="rgba(80, 80, 80, 0.5)",
                            landcolor="rgba(230, 230, 220, 1)",
                            showocean=True,
                            oceancolor="rgba(210, 230, 240, 1)",
                            showcountries=True,
                            countrycolor="rgba(120, 120, 120, 0.5)",
                        )
                    )
                except Exception as e:
                    logger.warning(f"Enhanced rendering not supported, falling back to standard rendering: {str(e)}")
        
        # Add the iftar zone as a filled area
        fig.add_trace(go.Scattergeo(
            lon=zone_lons,
            lat=zone_lats,
            mode='lines',
            fill='toself',
            line=dict(width=0),
            fillcolor='rgba(255, 140, 0, 0.3)',  # Semi-transparent orange fill, slightly more opaque
            hoverinfo='skip',
            name='Iftar Zone'
        ))
        
        # Add strong edges to the iftar zone
        fig.add_trace(go.Scattergeo(
            lon=band_edge_lons,
            lat=band_edge_lats,
            mode='lines',
            line=dict(
                width=3,
                color='rgba(255, 60, 0, 0.7)'  # Strong orange lines for the edges
            ),
            name='Iftar Zone Boundary'
        ))
        
        # Add center dotted line
        fig.add_trace(go.Scattergeo(
            lon=center_line_lons,
            lat=center_line_lats,
            mode='lines',
            line=dict(
                width=2,
                color='rgba(255, 120, 0, 0.7)',
                dash='dash'  # Dotted line
            ),
            name='Center of Iftar Zone'
        ))
        
        # Update statistics annotation with actual values
        stats_text = (
            f"<b>Statistics:</b><br>" +
            f"Muslim population breaking fast: <b>{int(total_population):,}</b><br>" +
            f"Longitude band: <b>{east_lon:.1f}° to {west_lon:.1f}°</b> ({window} minutes)"
        )
        
        # Update the first annotation (the statistics annotation)
        fig.layout.annotations[0].text = stats_text
    
    # Return total population info
    total_text = f"Total Muslim population breaking fast: {int(total_population):,}"
    
    # Return longitude range
    longitude_range = f"Longitude band: {east_lon:.1f}° to {west_lon:.1f}° ({window} minutes)"
    
    return fig, total_text, longitude_range, ""

# Animation control callback
@app.callback(
    [Output("animation-interval", "disabled"),
     Output("animation-button", "children"),
     Output("animation-running", "data")],
    [Input("animation-button", "n_clicks"),
     Input("globe-visualization", "relayoutData")],
    [State("animation-running", "data")],
    prevent_initial_call=True
)
def toggle_animation(n_clicks, relayout_data, is_running):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
    
    # If the user interacts with the map, stop the animation
    if 'globe-visualization.relayoutData' in trigger_id and is_running:
        logger.info("User interacted with map, stopping animation")
        return True, [html.I(className="fa fa-play"), " Animate"], False
    
    # If the animation button was clicked, toggle animation state
    if 'animation-button.n_clicks' in trigger_id and n_clicks:
        if is_running:
            logger.info("Animation stopped by button")
            return True, [html.I(className="fa fa-play"), " Animate"], False
        else:
            logger.info("Animation started by button")
            return False, [html.I(className="fa fa-pause"), " Pause"], True
    
    # Default: no change
    return no_update, no_update, no_update

# Time slider update callback for animation
@app.callback(
    Output("time-slider", "value"),
    [Input("animation-interval", "n_intervals")],
    [State("time-slider", "value"),
     State("animation-running", "data"),
     State("increment-slider", "value")],  # Add the increment slider value
    prevent_initial_call=True
)
def update_time_for_animation(n_intervals, current_time, is_running, time_increment):
    if not is_running:
        return no_update
    
    # Increment time by the selected increment amount for each interval
    new_time = (current_time + time_increment) % 1440  # Keep within 24 hours
    logger.info(f"Animation updating time to {new_time//60:02d}:{new_time%60:02d} (increment: {time_increment} min)")
    
    # Return the new time value
    return new_time

# Add a callback to handle viewstate changes - fix the duplicate output issue
@app.callback(
    Output("loading-output", "children", allow_duplicate=True),
    [Input("globe-visualization", "relayoutData")],
    prevent_initial_call=True
)
def track_view_state(relayout_data):
    if relayout_data:
        logger.debug(f"View state changed: {relayout_data}")
    return ""

# Add a callback to control zoom in standard geo mode
@app.callback(
    Output("globe-visualization", "figure", allow_duplicate=True),
    [Input("zoom-slider", "value")],
    [State("globe-visualization", "figure"),
     State("rendering-mode", "value")],
    prevent_initial_call=True
)
def update_zoom(zoom_level, current_fig, rendering_mode):
    if not current_fig:
        return no_update
        
    new_fig = go.Figure(current_fig)
    
    # Handle different rendering modes
    if rendering_mode == 'webgl':
        # For WebGL mode, update the camera distance
        if 'scene' in new_fig.layout and 'camera' in new_fig.layout.scene:
            camera = new_fig.layout.scene.camera
            
            # Get current eye position if available
            current_eye = camera.eye if hasattr(camera, 'eye') else {'x': 1.2, 'y': 0, 'z': 0}
            
            # Calculate direction vector
            mag = math.sqrt(current_eye.x**2 + current_eye.y**2 + current_eye.z**2)
            if mag > 0:
                # Normalize and scale by zoom level
                new_eye = {
                    'x': (current_eye.x / mag) * zoom_level,
                    'y': (current_eye.y / mag) * zoom_level,
                    'z': (current_eye.z / mag) * zoom_level
                }
                
                # Update camera
                new_fig.update_layout(
                    scene=dict(
                        camera=dict(
                            eye=new_eye,
                            # Keep other camera properties
                            center=camera.center if hasattr(camera, 'center') else {'x': 0, 'y': 0, 'z': 0},
                            up=camera.up if hasattr(camera, 'up') else {'x': 0, 'y': 0, 'z': 1}
                        )
                    )
                )
    else:
        # For standard mode, adjust the projection scale
        if 'geo' in new_fig.layout:
            # Standard mode uses orthographic projection
            # Adjust projection scale inversely (smaller value = more zoomed in)
            projection_scale = 1.2 / zoom_level  # Higher zoom = smaller scale value
            projection_scale = max(0.5, min(projection_scale, 2.0))  # Limit range
            
            new_fig.update_geos(
                projection_scale=projection_scale
            )
    
    return new_fig

# Run the application
if __name__ == '__main__':
    logger.info("Starting Iftar Globe Visualization app")
    # Initialize data at startup
    initialize_app_data()
    app.run_server(debug=True)

# Add a callback to store the increment value
@app.callback(
    Output("time-increment", "data"),
    [Input("increment-slider", "value")],
    prevent_initial_call=True
)
def store_time_increment(increment_value):
    logger.info(f"Setting time increment to {increment_value} minutes")
    return increment_value