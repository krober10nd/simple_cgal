import numpy as np
import cpputils as cutils

"""
Migration routines for moving points during parallel Delaunay
"""


def enqueue(blocks, points, faces, rank):
    """
    Return ranks that cell sites (vertices of triangulation) need to be sent
    """
    # determine llc (lower left corner) and urc (upper right corner)
    if rank == 0:
        nei_blocks = [blocks[rank + 1]]
    elif rank == len(blocks) - 1:
        nei_blocks = [blocks[rank - 1]]
    else:
        nei_blocks = [blocks[rank - 1], blocks[rank + 1]]

    le = np.array([np.amin(block, axis=0) for block in nei_blocks]).flatten()
    re = np.array([np.amax(block, axis=0) for block in nei_blocks]).flatten()

    # add dummy box if rank==0 or rank=size-1
    if rank == len(blocks) - 1:
        le = np.append(le, [-9999, -9999])
        re = np.append(re, [-9998, -9998])
    if rank == 0:
        le = np.insert(le, 0, [-9999, -9999])
        re = np.insert(re, 0, [-9998, -9998])

    exports = cutils.where_to2(points, faces, le, re)

    exports = np.where(exports == 1, rank + 1, exports)
    exports = np.where(exports == 0, rank - 1, exports)

    return exports
