import numpy as np
import pydetgen
from pydetgen.objects import ConfigBlock

block1 = ConfigBlock("block1", "array 1D", np.array([1, 2, 3]))
block2 = ConfigBlock("block2", "array 2D", np.array([[1, 2, 3], [4, 5, 6]]))
block3 = ConfigBlock(
    "block3", "object", [block1, block2, ConfigBlock("value1", "number", 66)]
)
print(pydetgen.fileio.yaml.dump_block(block1))
print(pydetgen.fileio.yaml.dump_block(block2))
print(pydetgen.fileio.yaml.dump_block(block3))
