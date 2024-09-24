import pydetgen

config = pydetgen.default_config()
config['detector']['']
# Write the yaml file
outfname = "test2.yaml"
pydetgen.yaml.write(config, outfname)

