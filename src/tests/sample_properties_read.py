import tomlkit
import pandas as pd

# Read the Excel file
df = pd.read_excel(r'C:\Users\shiau\QCAT\src\tests\template.xlsx',header=0)
print(df)
# Convert DataFrame to different dictionary formats
dict_by_columns = df.to_dict(orient='dict')
dict_by_rows = df.to_dict(orient='records')
dict_by_index = df.to_dict(orient='index')

# Display the results
print("Dictionary by columns:", dict_by_columns)
print("Dictionary by rows:", dict_by_rows)
print("Dictionary by index:", dict_by_index)


# Assuming 'config.toml' is your file
# with open(r'src/tests/sample_properties.toml', 'r') as file:
#     content = file.read()
#     data = tomlkit.parse(content)

# print(type(data))
# print(data)

# print(data["operation"])
# print(data["operation"][0]["parameters"])

# print(data)