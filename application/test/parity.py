import os
import zipfile
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Function to extract all .nc files from a zip archive
def extract_nc_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to generate histograms for all .nc files in a directory

def determine_y_range(input_dir, bins=50, x_range=None):
    """
    Determine the common y-axis range for histograms by finding the maximum frequency across all data.
    """
    max_frequency = 0

    for root, _, files in os.walk(input_dir):
        if "__MACOSX" in root:  # Skip the __MACOSX folder
            continue

        for file in files:
            if file.endswith('.nc') and not file.startswith("._"):  # Skip hidden macOS files
                file_path = os.path.join(root, file)
                try:
                    dataset = xr.open_dataset(file_path, engine="netcdf4")
                    variable_name = list(dataset.data_vars.keys())[0]
                    data = dataset[variable_name].values.flatten()

                    # Calculate histogram frequencies
                    hist, _ = np.histogram(data, bins=bins, range=x_range)
                    max_frequency = max(max_frequency, hist.max())
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    return (0, max_frequency)


def generate_histograms(input_dir, output_dir, bins=50, x_range=None, y_range=None):
    """
    Generate histograms for all .nc files in a directory and save them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        if "__MACOSX" in root:  # Skip the __MACOSX folder
            continue

        for file in files:
            if file.endswith('.nc') and not file.startswith("._"):  # Skip hidden macOS files
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    dataset = xr.open_dataset(file_path, engine="netcdf4")
                    variable_name = list(dataset.data_vars.keys())[0]
                    data = dataset[variable_name].values.flatten()

                    # Generate histogram
                    plt.figure(figsize=(8, 6))
                    plt.hist(data, bins=bins, range=x_range, edgecolor='k', alpha=0.7)
                    plt.title(f"Histogram of {variable_name} ({file})")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.ylim(y_range)  # Fix the y-axis range
                    plt.grid(True)

                    # Save histogram
                    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_histogram.png")
                    plt.savefig(output_path)
                    plt.close()
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


# Main execution
if __name__ == "__main__":
    zip_file_path = r"c:\Users\shiau\OneDrive\文件\H18M55S43-130.zip"  # Replace with the actual path
    extract_folder = r"D:\data\raw"  # Replace with your desired folder
    output_folder = r"D:\data\plot"  # Replace with your desired output folder

    # Extract .nc files from the zip archive
    extract_nc_files(zip_file_path, extract_folder)

    # Determine common x_range and y_range
    first_nc_file = None
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith('.nc'):
                first_nc_file = os.path.join(root, file)
                break
        if first_nc_file:
            break

    if first_nc_file:
        dataset = xr.open_dataset(first_nc_file)
        variable_name = list(dataset.data_vars.keys())[0]
        data = dataset[variable_name].values.flatten()
        x_range = (np.min(data), np.max(data))
    else:
        x_range = None  # Use automatic range if no files are found

    y_range = determine_y_range(extract_folder, bins=50, x_range=x_range)

    # Generate histograms for all .nc files
    generate_histograms(extract_folder, output_folder, bins=50, x_range=x_range, y_range=y_range)

