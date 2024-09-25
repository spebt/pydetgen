import spebtschema
import yaml

# load the YAML file
with open('test2.yaml', 'r') as f:
    config = yaml.safe_load(f)
    try:
        spebtschema.yaml.validate(config, name='main')
    except Exception as e:
        raise e