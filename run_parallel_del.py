import numpy as np

import simple_cgal
import utils


num_points = 100
points = np.random.random((num_points, 2))
faces = simple_cgal.delaunay2(points[:, 0], points[:, 1])
cc, rr = utils.circumballs(points, faces)
utils.plot_circumballs(points, faces, cc, rr)
