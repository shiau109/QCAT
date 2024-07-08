
import os

# Specify the directory path
folder_path = r'D:\Data\5Q4C_0411_3_DR4\iSWAP_34'
print(os.listdir(folder_path))
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# Print the list of files
print(file_names)