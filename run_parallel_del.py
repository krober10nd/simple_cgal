"""
WIP script that will eventually become the
driver for the parallel delaunay algorithm.
"""
import numpy as np
from mpi4py import MPI

import simple_cgal
import utils

import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

num_blocks = size
num_points = 100000
gpoints = np.random.random((num_points, 2))

t1 = time.time()
block_sets = utils.blocker(gpoints, num_blocks)
print("block time is ", str(time.time() - t1), flush=True)
points = block_sets[rank]
t1 = time.time()
faces = simple_cgal.delaunay2(points[:, 0], points[:, 1])
print("tria time is ", str(time.time() - t1), flush=True)
t1 = time.time()
toMigrate = utils.enqueue(block_sets, points, faces, rank)
print("enqueue time is ", str(time.time() - t1), flush=True)

quit()
# check
if rank == 0:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.collections

    fig, ax = plt.subplots()
    plt.plot(points[:, 0], points[:, 1], "r.")
    plt.triplot(points[:, 0], points[:, 1], faces, color="g")
    cc, rr = utils.calc_circumballs(points, faces,)
    patches1 = [
        plt.Circle(center, size, fill=None, color="black")
        for center, size in zip(cc, rr)
    ]
    coll1 = matplotlib.collections.PatchCollection(patches1, match_original=True,)
    ax.add_collection(coll1)

    for ix in toMigrate:
        plt.plot(points[ix[0], 0], points[ix[0], 1], "bs")
        for l in ix[1::]:
            plt.text(points[ix[0], 0], points[ix[0], 1], str(l))

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
    plt.show()
