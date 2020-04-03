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
    vertices = [v for v in vtoe]
    vcc = [cc[v] for v in vertices]
    vrr = [rr[v] for v in vertices]
    t1 = time.time()
    nIntrscts, blockNums = utils.which_intersect(block_sets, vcc, vrr, rank)
    print("Intersect time is " + str(time.time() - t1), flush=True)

    exports = []
    # t1 = time.time()
    # kount2 = 0
    # kount = 0
    # exports = []
    # whereTo = []
    # for cid in range(len(points)):
    #    tmpWhereTo = []
    #    vertices = vtoe[cid, 0 : nne[cid]]
    #    for vertex in vertices:
    #        if num_intersects[kount2] > 0:
    #            # this cell site should be exported to block #'s in whereTo
    #            tmpWhereTo.append(block_nums[kount2])
    #        if np.sum(num_intersects[kount2]) > 0:
    #            # remove duplicate entries in tmp arrays
    #            whereTo = np.unique([[j for j in i if j > -1] for i in tmpWhereTo])
    #            exports.append([])
    #            exports[kount] = [cid, whereTo]
    #            kount += 1
    #        kount2 += 1
    # print("Loading time is " + str(time.time() - t1), flush=True)
    return exports
