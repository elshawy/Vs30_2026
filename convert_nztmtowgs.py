import pandas as pd
from pyproj import Transformer

def convert_nztm_to_wgs84(input_csv_path, output_csv_path):
    """
    Reads a CSV file containing NZTM 2193 coordinates (easting, northing),
    converts them to WGS84 (longitude, latitude), and saves the result to a new CSV.
    
    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to save the output CSV file.
    """
    # Load the CSV file into a pandas DataFrame
    # Assuming the input CSV has columns named 'easting' and 'northing'
    df = pd.read_csv(input_csv_path)
    
    # Check if required columns exist in the DataFrame
    if not {'easting', 'northing'}.issubset(df.columns):
        raise ValueError("The CSV file must contain 'easting' and 'northing' columns.")

    # Initialize the pyproj Transformer
    # EPSG:2193 represents NZTM 2000 (New Zealand Transverse Mercator 2000)
    # EPSG:4326 represents WGS84
    # always_xy=True ensures the coordinate order is (lon, lat) instead of (lat, lon)
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)

    # Perform the coordinate transformation
    # transformer.transform accepts arrays, making it efficient for pandas columns
    lon, lat = transformer.transform(df['easting'].values, df['northing'].values)

    # Assign the transformed coordinates to new columns in the DataFrame
    df['longitude'] = lon
    df['latitude'] = lat

    # Save the updated DataFrame to a new CSV file
    # index=False prevents pandas from writing row indices to the file
    df.to_csv(output_csv_path, index=False)
    print(f"Conversion successful! Output saved to: {output_csv_path}")

# Example usage:
# Replace 'input_data.csv' and 'output_data.csv' with your actual file paths
if __name__ == "__main__":
    INPUT_FILE = "measured_sites.csv"   # Insert your input CSV file name here
    OUTPUT_FILE = "measured_sites_converted.csv" # Insert your desired output CSV file name here
    
    # Call the conversion function
    convert_nztm_to_wgs84(INPUT_FILE, OUTPUT_FILE)
