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
    import time

    cc, rr = utils.calc_circumballs(points, faces)
    vtoe, nne = utils.vertex_to_elements(faces)
    t1 = time.time()
    blockNums = utils.which_intersect(block_sets, cc, rr, rank)
    print("Intersect time is " + str(time.time() - t1), flush=True)
    for num in blockNums:
        print(num)
    # now go through each voronoi cell
    exports = []
    return exports
