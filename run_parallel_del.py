import time

import numpy as np
from mpi4py import MPI
from scipy.spatial import Delaunay

import utils

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

np.random.seed(0)
gpoints = np.random.random((1000000, 2))
# gpoints = np.loadtxt("points.out", delimiter=",")

if rank == 0:
    print("read it in", flush=True)

t1 = time.time()

points, extents = utils.blocker(points=gpoints, rank=rank, nblocks=size)

tria = Delaunay(points, incremental=True)

exports = utils.enqueue(extents, points, tria.vertices, rank, size)

new_points = utils.migration(comm, rank, size, exports)

tria.add_points(new_points, restart=True)

faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])

points, faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])

upoints, ufaces = utils.aggregate(points, faces, comm, size, rank)

comm.barrier()
if rank == 0:
    print("finished in " + str(time.time() - t1))
    np.savetxt("faces" + str(rank), ufaces)
    np.savetxt("points" + str(rank), upoints)
