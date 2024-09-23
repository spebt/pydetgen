import yaml
import numpy as np

detconfig = {
    "detector": {
        "detector geometry": [
            [0.0, 3, 0, 2, -0.5, 0.5, 0, 1.0],
            [0.0, 3, 2.5, 4.5, -0.5, 0.5, 0, 1.0],
            [0.0, 3, 5, 7, -0.5, 0.5, 0, 1.0],
        ],
        "active detector": [1]
    }
}

print(yaml.dump(detconfig, default_flow_style=None,line_break=True))
