import numpy as np
from mpi4py import MPI
import time

import utils
import simple_cgal as cgal


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

bbox = (0, 1, 0, 1)
bbox = np.array(bbox).reshape(-1, 2)
h0 = 0.0005
gpoints = np.mgrid[tuple(slice(min, max + h0, h0) for min, max in bbox)]
gpoints = gpoints.reshape(2, -1).T
if rank == 0:
    print("domain has: " + str(len(gpoints)))


if size > 1:
    if rank == 0:
        t1 = time.time()
    blocks = utils.blocker(gpoints, size)
    points = blocks[rank]
    faces = cgal.delaunay2(points[:, 0], points[:, 1])
    exports = utils.enqueue(blocks, points, faces, rank)
    t2 = time.time()
    new_points = utils.migration(comm, rank, exports, points)
    print(time.time() - t2)
    points = np.append(points, new_points, axis=0)
    faces = cgal.delaunay2(points[:, 0], points[:, 1])
    if rank == 0:
        print("Parallel time is " + str(time.time() - t1))
else:
    t1 = time.time()
    faces = cgal.delaunay2(gpoints[:, 0], gpoints[:, 1])
    print("Serial time is " + str(time.time() - t1))
