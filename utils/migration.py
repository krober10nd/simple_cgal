import utils

"""
Migration routines for moving points during parallel
Delaunay
"""


def enqueue(block_sets, points, faces, rank):
    """
    Return cell sites (vertices of triangulation) to be
    sent to blocks in `whereTo` that intersects with their circumballs.
    """

    cc, rr = utils.calc_circumballs(points, faces)
    vtoe, nne = utils.vertex_to_elements(faces)
    blockNums = utils.which_intersect(block_sets, cc, rr, rank)
    # now go through each Voronoi cell
    # if cell has a vertex with a circumball that intersects
    # with a neighboring block, flag point to be sent to that block.
    return blockNums
