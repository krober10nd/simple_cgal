"""
WIP script that will eventually become the
driver for the parallel delaunay algorithm.

all local quantitis are pre-fixed with l_
all global quantites are pre-fixed with g_
"""
import numpy as np
from mpi4py import MPI

import simple_cgal
import utils

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

g_num_points = 100
g_num_blocks = size
g_points = np.random.random((g_num_points, 2))

# divide all input points into blocks
g_block_sets = utils.blocker(g_points, g_num_blocks)
l_points = g_block_sets[rank]
# compute Delaunay triangulation of input point in block
l_faces = simple_cgal.delaunay2(l_points[:, 0], l_points[:, 1])
# compute the circumballs of each triangle in block
l_cc, l_rr = utils.calc_circumballs(l_points, l_faces)
# determine which block(s) a circumball intersects with
l_num_intersects, l_block_nums = utils.which_intersect(g_block_sets, l_cc, l_rr, rank)
# determine which points are finite (and infinite)
l_finitePoints = l_points[np.where(utils.are_finite(l_points))]
l_infinitePoints = l_points[np.where(np.invert(utils.are_finite(l_points)))]


import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.patches as patches

if rank == 0:
    fig, ax = plt.subplots()
    plt.triplot(l_points[:, 0], l_points[:, 1], l_faces, c="#FFAC67")
    plt.plot(l_finitePoints[:, 0], l_finitePoints[:, 1], "r.")
    plt.plot(l_infinitePoints[:, 0], l_infinitePoints[:, 1], "b.")
    plt.show()
    quit()
    patches1 = [
        plt.Circle(center, size, fill=None, color="red")
        for center, size, block_num in zip(l_cc, l_rr, l_block_nums)
        if np.any(l_block_num[0] != rank)
    ]
    coll1 = matplotlib.collections.PatchCollection(patches1, match_original=True,)
    ax.add_collection(coll1)

    # plot blocks
    for block in block_sets:
        le = np.amin(block, axis=0)
        re = np.amax(block, axis=0)
        rect = patches.Rectangle(
            (le[0], le[1]),
            re[0] - le[0],
            re[1] - le[1],
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set(xlim=(0, 1), ylim=(0, 1))

# plt.show()
