# PyDetgen

SPEBT project Python package for generating the detector system configuration file 

## Get Started

### Install

```shell
pip install .
```
### Example

#### Generate configuration file

```python
import pydetgen

# Get default configuration
config = pydetgen.default_config()

# Modify the configuration
det_pos = np.array()
det_geoms = get_det_geoms(0.475, np.array([[1.2, 8, 0]]), np.array([3, 2, 1]))
plate_geoms = get_plate_geoms_2d(3.5, 0, 16, np.array([2]), np.array([8]), 0, 1, 1)
config['detector']['detector geometry'] = 

# Write the configuration to a yaml file
outfname = "minimal_test.yaml"
pydetgen.yaml.write(config, outfname)
```

#### Validate the configuration with spebt-schema package

```python
import spebtschema
import yaml

# load the YAML file
with open('minimal_test.yaml', 'r') as f:
    config = yaml.safe_load(f)
    try:
        spebtschema.yaml.validate(config, name='main.json')
    except Exception as e:
        raise e
```