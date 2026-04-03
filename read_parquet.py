import pandas as pd

def convert_parquet_to_csv(parquet_file_path, csv_file_path):
    """
    Reads a Parquet file and saves its contents as a CSV file.
    """
    try:
        # ---------------------------------------------------------
        # 1. Read the Parquet file
        # ---------------------------------------------------------
        # Load the data from the specified Parquet file into a Pandas DataFrame.
        # Note: You may need 'pyarrow' or 'fastparquet' installed in your environment.
        df = pd.read_parquet(parquet_file_path)
        
        # ---------------------------------------------------------
        # 2. Save the DataFrame to a CSV file
        # ---------------------------------------------------------
        # Export the DataFrame to a CSV file.
        # Setting 'index=False' prevents Pandas from writing row numbers as the first column.
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        
        # Output a success message if the process completes without errors.
        print(f"[Success] Successfully converted '{parquet_file_path}' to '{csv_file_path}'.")
        
    except FileNotFoundError:
        # Handle the error if the source Parquet file does not exist.
        print(f"[Error] The file '{parquet_file_path}' was not found. Please check the path.")
    except Exception as e:
        # Catch and print any other unexpected errors during the process.
        print(f"[Error] An unexpected error occurred: {e}")

# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Define the input and output file names
    input_file = 'stations.parquet'
    output_file = 'stations_Fe.csv'
    
    # Run the conversion function
    convert_parquet_to_csv(input_file, output_file)
