
import numpy as np

crosstalk = np.array(
    [[1., -0.036, -0.013],
    [0.0, 1., 0.0],
    [0.029, 0.048, 1.],
    ])
cancel = np.linalg.inv(crosstalk)
print(cancel)