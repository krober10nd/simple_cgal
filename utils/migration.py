import numpy as np
import time

import utils

"""
Migration capabilities for moving points during parallel
Delaunay
"""


def enqueue_pass1(block_sets, points, faces, rank):
    """
    1. Enqueue finite sites that are near to block boundaries
        to any neighbors within circumball radii
    2. Enqueue inifite sites to single closest neighbor
        distance is based on centroid of blocks and point coordinates.

    PointsToExport has the form of:
    points ID, block_i, block_i+1,..., block_n
    """
    FinitePointsToExport = __enqueue_finite(block_sets, points, faces, rank)
    # exp_inf_points, whereTo = __enqueue_infinite()
    # exported = exp_finite_points + exp_inf_points
    PointsToExport = FinitePointsToExport  # + InfinitePointsToExport
    return PointsToExport


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


def __enqueue_infinite():
    """
    Export infinite points to closest neighboring block.
    """
    return 0
