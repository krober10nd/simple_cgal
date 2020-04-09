from mpi4py import MPI
import numpy as np
from scipy.spatial import Delaunay

import utils

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


gpoints = np.loadtxt("points.out", delimiter=",")

points, extents = utils.blocker(points=gpoints, rank=rank, nblocks=size)

tria = Delaunay(points, incremental=True)

exports = utils.enqueue(extents, points, tria.vertices, rank, size)

new_points = utils.migration(comm, rank, size, exports)

tria.add_points(new_points)

points = np.append(points, new_points, axis=0)

# this is horribly slow will have to rewrite this
faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.collections

if rank == 1:
    fig, ax = plt.subplots()
    for e in extents:
        rect = matplotlib.patches.Rectangle(
            (e[0], e[1]), e[2] - e[0], e[3] - e[1], edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)

    plt.triplot(points[:, 0], points[:, 1], faces)

    # patches = [plt.Circle(center, size, fill=None) for center, size in zip(cc, rr)]
    # coll = matplotlib.collections.PatchCollection(patches, match_original=True,)
    # ax.add_collection(coll)

    # ax.set_xlim([-11000, -8700])
    # ax.set_ylim([32500, 34600])
    ax.axis("equal")
    plt.show()
