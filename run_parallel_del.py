"""
WIP script that will eventually become the
driver for the parallel delaunay algorithm.
"""
import numpy as np

import simple_cgal
import utils


num_points = 100
num_blocks = 2
points = np.random.random((num_points, 2))

# compute Delaunay triangulation of input point set
faces = simple_cgal.delaunay2(points[:, 0], points[:, 1])
# compute the circumballs of each triangle
cc, rr = utils.calc_circumballs(points, faces)
# divide all input points into blocks
block_sets = utils.blocker(points, num_blocks)
# determine which block(s) a circumcircle intersects with
num_intersects, block_nums = utils.which_intersect(block_sets, cc, rr)


# testing if its correct
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.patches as patches

fig, ax = plt.subplots()
# plt.triplot(points[:, 0], points[:, 1], faces , c="#FFAC67")
patches1 = [
    plt.Circle(center, size, fill=None, color="red")
    for center, size, block_num in zip(cc, rr, block_nums)
    if block_num[1] == 0
]
patches2 = [
    plt.Circle(center, size, fill=None, color="black")
    for center, size, block_num in zip(cc, rr, block_nums)
    if block_num[1] == 1
]

coll1 = matplotlib.collections.PatchCollection(patches1, match_original=True,)
coll2 = matplotlib.collections.PatchCollection(patches2, match_original=True,)
ax.add_collection(coll1)
ax.add_collection(coll2)

for block in block_sets:
    le = np.amin(block, axis=0)
    re = np.amax(block, axis=0)
    print(le, re)
    rect = patches.Rectangle(
        (le[0], le[1]), re[0] - le[0], re[1] - le[1], edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)

plt.show()
