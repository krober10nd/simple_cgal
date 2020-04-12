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

points = np.append(points, new_points, axis=0)


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.collections

if rank == 1:
    fig, ax = plt.subplots()
    plt.plot(new_points[:, 0], new_points[:, 1], "r.")
    for e in extents:
        rect = matplotlib.patches.Rectangle(
            (e[0], e[1]), e[2] - e[0], e[3] - e[1], edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
>>>>>>> ce77101757c8825e7a6e0ed5800dd42f8dd33526

points, faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])

upoints, ufaces = utils.aggregate(points, faces, comm, size, rank)

comm.barrier()
if rank == 0:
    print("finished in " + str(time.time() - t1))
    np.savetxt("faces" + str(rank), ufaces)
    np.savetxt("points" + str(rank), upoints)
