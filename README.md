# Ramadan Fast Visualization: Global Iftar Zone Tracker

# Detailed Workflow Description: Estimating and Visualizing the Fasting Muslim Population (Iftar Zone)

An interactive visualization tool that shows the global Iftar zone (where Muslims break their fast during Ramadan) in real-time. The application displays Muslim population distribution and tracks the sunset line around the globe, highlighting those areas where the fast is currently ending.

![Iftar Zone Visualization Screenshot](screenshot.jpg)

## Features

- **Real-time Iftar Zone Tracking**: Visualizes the global geographical area where Muslims are breaking their fast based on sunset times.
- **Population Heatmap**: Shows Muslim population density with a color gradient to highlight areas with significant fasting populations.
- **Interactive Controls**:
  - Time slider to set specific times
  - Window width adjustment for Iftar zone
  - Animation with customizable time increment
  - Zoom level control
  - Toggle between standard and WebGL 3D rendering modes
- **Detailed Statistics**: Displays the total Muslim population currently breaking their fast.
- **Smooth Animations**: Track the Iftar zone as it moves across the globe in real-time.

## Data Sources & Processing

## 1. Data Sources & Acquisition

### Population Density Data
- **Source:**  
  Gridded Population of the World, Version 4 (GPWv4): Population Density Adjusted to Match 2015 Revision UN WPP Country Totals, Revision 11 (Version 4.11)
- **Citation:**  
  Center For International Earth Science Information Network-CIESIN-Columbia University. (2018). *Gridded Population of the World, Version 4 (GPWv4): Population Density Adjusted to Match 2015 Revision UN WPP Country Totals, Revision 11 (Version 4.11) [Data set].* Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC). [https://doi.org/10.7927/H4F47M65](https://doi.org/10.7927/H4F47M65)

### World Boundary Data
- **Source:**  
  World Administrative Boundaries obtained from OpenDataSoft.
- **Citation:**  
  World Administrative Boundaries. Retrieved from [https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/?dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6IndvcmxkLWFkbWluaXN0cmF0aXZlLWJvdW5kYXJpZXMiLCJvcHRpb25zIjp7fX0sImNoYXJ0cyI6W3siYWxpZ25Nb250aCI6dHJ1ZSwidHlwZSI6ImNvbHVtbiIsImZ1bmMiOiJDT1VOVCIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNGRjUxNUEifV0sInhBeGlzIjoic3RhdHVzIiwibWF4cG9pbnRzIjo1MCwic29ydCI6IiJ9XSwidGltZXNjYWxlIjoiIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D&location=2,41.81735,0.00845&basemap=jawg.light).

### Muslim Population Percentages

- **Sources:**  
  Muslim population percentages were compiled from reputable sources including ARDA, the CIA World Factbook, and Pew Research Center.
- **Note on Granularity:**  
  The Muslim population percentages are available at the country level. While state- or district-level data would yield more accurate local estimates, such data is not available globally. Therefore, our analysis uses country boundaries, which provides a broad overview.
- **Adjustment**: Applied a 75% fasting rate to account for non-fasting Muslims (children, ill, pregnant women, travelers, etc.)

## Detailed Workflow

### 2. Geoprocessing in ArcGIS Pro

#### Step 1: Enriching the World Boundary with Muslim Population Percentages
- **Process:**
  - The world administrative boundaries feature class was enriched by adding a column for Muslim population percentages.
  - These percentages were sourced from ARDA, CIA World Factbook, and Pew Research Center.
  - **Adjustment:**  
    Since not all Muslims fast (e.g., children, pregnant women, and other exempt groups), a fasting rate of 75% (i.e., subtracting 25%) was applied. For example, if a country has 80% Muslims, only 80 × 0.75 = 60% are considered fasting.

#### Step 2: Converting the Enriched Feature Class to a Raster
- **Tool:**  
  The "Feature to Raster" tool in ArcGIS Pro was used to convert the enriched feature class to a raster.
- **Purpose:**  
  This step ensured that the Muslim fasting percentage is in raster format, matching the spatial resolution and extent of the population density raster.

#### Step 3: Combining with Population Density Data
- **Process:**
  - The fasting percentage raster (now representing the percentage of Muslims who fast) was multiplied by the GPWv4 population density raster.
  - **Calculation:**  
    ```python
    "Fasting_Percentage_Raster" * ("Population_Density_Raster" / 100)
    ```
    This converts the percentage to decimal form and yields a raster representing the actual number of fasting Muslims per pixel.

#### Step 4: Resolution Management
- Created both high-resolution (30 arc-second) and low-resolution versions for performance optimization.
- The low-resolution raster was created using aggregation in ArcGIS Pro to improve rendering speed.

### 3. Python Implementation

The application is built with Python using:

- **Dash & Plotly**: For interactive web-based visualization
- **Rasterio**: For working with geospatial raster data
- **NumPy**: For efficient numerical computations
- **WebGL**: For 3D globe rendering with hardware acceleration

Key implementation components:

1. **Iftar Zone Calculation**:
   - Determines the longitude band where sunset is occurring.
   - Uses the Earth's rotation rate of 15° per hour (0.25° per minute).
   - Adjusts for a user-defined time window (default: 10 minutes).

2. **Visualization Modes**:
   - **Standard Mode**: 2D orthographic globe projection with country boundaries.
   - **WebGL 3D Mode**: Interactive 3D globe with cosmic background and enhanced visual effects.

3. **Performance Optimizations**:
   - Cached data loading to minimize read operations.
   - Efficient filtering of visible points using vector operations.
   - Selective rendering based on camera position to reduce point count.
   - Pre-calculation of cartesian coordinates for 3D rendering.

## Installation and Usage

### Prerequisites

- Python 3.8+
- Required Python packages (see requirements.txt)
- Downloaded raster data files in the correct location

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ramadan-fast-visualization.git
   cd ramadan-fast-visualization
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure raster files are in the `script/raster/` directory:
   - `Total_Muslim_Fasting_Population.tif` (high resolution)
   - `tot_pop_mus_fast_low_res.tif` (low resolution)

### Running the Application

Run the application with:
```
python script/iftar_globe_visualization.py
```

The application will start a local web server, accessible at http://127.0.0.1:8050/

### Controls

- **Time Slider**: Set the current UTC time.
- **Window Slider**: Adjust the width of the Iftar window (in minutes).
- **Time Increment Slider**: Control how quickly time advances during animation.
- **Zoom Level Slider**: Adjust the view distance from the globe.
- **Animation Button**: Start/pause the animation of the Iftar zone.
- **Recenter Button**: Reset the view to the current Iftar zone.
- **Rendering Mode**: Toggle between standard and WebGL 3D mode.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data providers: CIESIN Columbia University, ARDA, CIA World Factbook, and Pew Research Center
- OpenDataSoft for administrative boundary data
- The Dash and Plotly communities for visualization libraries 