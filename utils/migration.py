import utils
import cpputils as cutils

"""
Migration routines for moving points during parallel Delaunay
"""


def enqueue(block_sets, points, faces, rank):
    """
    Return cell sites (vertices of triangulation) to be
    sent to blocks in `export` that intersect with their circumballs.
    """

    cc, rr = utils.calc_circumballs(points, faces)
    vtoe, nne = utils.vertex_to_elements(faces)
    intersects = utils.which_intersect(block_sets, cc, rr, rank)
    export = cutils.where_to(intersects, vtoe, rank)
    return export
