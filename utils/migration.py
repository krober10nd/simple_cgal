import numpy as np

import utils

"""
Migration routines for moving points during parallel
Delaunay
"""


def enqueue_pass1(block_sets, points, faces, rank):
    """
    1. Enqueue finite sites that are near to block boundaries
        to any neighbors within circumball radii
    2. Enqueue inifite sites to single closest neighbor
        distance is based on centroid of blocks and point coordinates.

    PointsToExport has the form of:
        points ID,block_i,block_i+1,...,block_n
    """
    FinitePointsToExport = __enqueue_finite(block_sets, points, faces, rank)
    InfinitePointsToExport = __enqueue_infinite_pass1(block_sets, points, rank)
    return FinitePointsToExport + InfinitePointsToExport


def __enqueue_finite(block_sets, points, faces, rank):
    """
    Return finite cell sites (vertices of triangulation) to be
    sent to blocks in `whereTo` that intersects with their circumballs.
    """
    cc, rr = utils.calc_circumballs(points, faces)
    areFiniteCells = utils.are_finite(points)
    # calc vertices of each voronoi cell
    vtoe, nne = utils.vertex_to_elements(faces)
    ncc = []
    nrr = []
    exports = []
    whereTo = []
    for cid in range(len(points)):
        if areFiniteCells[cid]:
            # vertices of voronoi cell
            vertices = vtoe[0 : nne[cid], cid]
            for vertex in vertices:
                ncc.append(cc[vertex])
                nrr.append(rr[vertex])

    num_intersects, block_nums = utils.which_intersect(block_sets, ncc, nrr, rank)
    kount2 = 0
    kount = 0
    for cid in range(len(points)):
        if areFiniteCells[cid]:
            tmpWhereTo = []
            vertices = vtoe[0 : nne[cid], cid]
            for vertex in vertices:
                if num_intersects[kount2] > 0:
                    # this cell site should be exported to block #'s in whereTo
                    tmpWhereTo.append(block_nums[kount2])
                if np.sum(num_intersects[kount2]) > 0:
                    # remove duplicate entries in tmp arrays
                    whereTo = np.unique([[j for j in i if j > -1] for i in tmpWhereTo])
                    exports.append([])
                    exports[kount] = [cid, whereTo]
                    kount += 1
                kount2 += 1
    return exports


def __enqueue_infinite_pass1(block_sets, points, rank):
    """
    Export infinite points to closest neighboring block.
    """
    # determine which points are infinite
    areInfiniteCells = np.invert(utils.are_finite(points))

    # calculate centroid of block
    centroid = []
    for block in block_sets:
        le = np.amin(block, axis=0)
        re = np.amax(block, axis=0)
        centroid.append((le + re) / 2)

    exports = []
    # measure distance from block # - 1 and block # + 1
    ix = 0
    kount = 0
    for point, isInfiniteCell in zip(points, areInfiniteCells):
        if isInfiniteCell:
            dst1 = np.inf
            dst2 = np.inf
            if rank != 0:
                dst1 = (point[0] - centroid[rank - 1][0]) ** 2 + (
                    point[1] - centroid[rank - 1][1]
                ) ** 2
            if rank != len(block_sets) - 1:
                dst2 = (point[0] - centroid[rank + 1][0]) ** 2 + (
                    point[1] - centroid[rank + 1][1]
                ) ** 2
            exports.append([])
            if dst1 < dst2:
                exports[kount] = [ix, (rank - 1)]
            else:
                exports[kount] = [ix, (rank + 1)]
            kount += 1
        ix += 1
    return exports
