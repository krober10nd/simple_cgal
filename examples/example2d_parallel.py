import time
import numpy as np
from mpi4py import MPI
from scipy.spatial import Delaunay as SciPyDelaunay
import matplotlib.pyplot as plt

import utils

num_points = 250000
np.random.seed(0)
gpoints = np.random.uniform(size=(num_points, 2), low=0.0, high=1.0)
bbox = (0.0, 0.0, 1.0, 1.0)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t1 = time.time()

points, extents = utils.blocker(points=gpoints, rank=rank, nblocks=size, bbox=bbox)

tria = SciPyDelaunay(points, incremental=True)

exports = utils.enqueue(extents, points, tria.vertices, rank, size)

new_points = utils.migration(comm, rank, size, exports)

tria.add_points(new_points, restart=True)

points, faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])

upoints, ufaces = utils.aggregate(points, faces, comm, size, rank)

comm.barrier()
if rank == 0:
    print("elapsed time is " + str(time.time() - t1), flush=True)
    print(upoints.shape)
    print(ufaces.shape)
    print(ufaces[:5, :])
    plt.triplot(upoints[:, 0], upoints[:, 1], ufaces)
    plt.show()
