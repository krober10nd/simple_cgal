import numpy as np
from mpi4py import MPI
import time

import utils

from scipy.spatial import Delaunay

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# bbox = (0, 2, 0, 2)
# bbox = np.array(bbox).reshape(-1, 2)
# h0 = 0.005
# gpoints = np.mgrid[tuple(slice(min, max + h0, h0) for min, max in bbox)]
# gpoints = gpoints.reshape(2, -1).T

gpoints = np.loadtxt("points.out", delimiter=",")
if rank == 0:
    print("domain has # of points: " + str(len(gpoints)), flush=True)


if size > 1:
    t1 = time.time()

    # decompose points into blocks
    blocks = utils.blocker(gpoints, size)
    points = blocks[rank]

    # triangulate local points
    tria = Delaunay(points, incremental=True, qhull_options="")

    np.savetxt("b4points" + str(rank), points, delimiter=",")
    np.savetxt("b4faces" + str(rank), tria.vertices, delimiter=",")

    # determine which points to export
    exports = utils.enqueue(blocks, points, tria.vertices, rank)

    # migrate these points
    new_points = utils.migration(comm, rank, size, exports)

    # augmented triangulation incrementally
    tria.add_points(new_points)

    print("Parallel time is " + str(time.time() - t1), flush=True)

    points = np.append(points, new_points, axis=0)
    np.savetxt("addedpoints" + str(rank), new_points, delimiter=",")
    np.savetxt("points" + str(rank), points, delimiter=",")
    np.savetxt("faces" + str(rank), tria.vertices, delimiter=",")
else:
    t1 = time.time()
    tria = Delaunay(gpoints)
    print("Serial time is " + str(time.time() - t1))
