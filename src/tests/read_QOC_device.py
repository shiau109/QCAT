import yaml

# Replace 'your_file.yaml' with the path to your YAML file
with open(r'C:\Users\shiau\QCAT\src\tests\quantrolOx_device_config.yaml', 'r') as file:
    data = yaml.safe_load(file)

print(data["components"])