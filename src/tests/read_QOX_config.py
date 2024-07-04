import tomlkit

# Assuming 'config.toml' is your file
with open(r'src/tests/quantrolOx_config.toml', 'r') as file:
    content = file.read()
    data = tomlkit.parse(content)

print(data)