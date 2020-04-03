import numpy as np

import cpputils as cutils


"""
Utilities for the parallel Delaunay algorithm.
"""


def vertex_to_elements(faces):
    """
    Returns the elements incident to a vertex in the
    Delaunay graph.

    faces: an ndarray of int, `(ndarray of ints, shape (nsimplex, ndim+1)`.
            Indices of the points forming the simplices in the triangulation.
    """
    num_points = np.amax(faces)+1
    num_faces = len(faces)
    vtoe = cutils.vertex_to_elements(faces, num_points, num_faces)
    nne = np.count_nonzero(vtoe, axis=1)
    return vtoe, nne


def simple_which_intersect(block_sets, circumcenters, radii, rank):
    num_trias = len(circumcenters)
    ndim = len(circumcenters[0])

    intersects = np.zeros((num_trias, 5), dtype=int) - 1
    num_intersects = np.zeros((num_trias, 1), dtype=int)

    if rank == 0:
        nei_blocks = [block_sets[rank + 1]]
    elif rank == len(block_sets) - 1:
        nei_blocks = [block_sets[rank - 1]]
    else:
        nei_blocks = [block_sets[rank - 1], block_sets[rank + 1]]

    le = []
    re = []
    for block in nei_blocks:
        le.append(np.amax(block, axis=0))
        re.append(np.amax(block, axis=0))

    for tria, (cc, rr) in enumerate(zip(circumcenters, radii)):
        for block_num, block in enumerate(nei_blocks):
            if __do_intersect(ndim, cc, rr, le[block_num], re[block_num]):
                intersects[tria, num_intersects[tria]] = block_num
                num_intersects[tria] += 1
    return num_intersects, intersects


def which_intersect(block_sets, circumcenters, radii, rank):
    """
    Returns a list with the block # the circumball intersects with.
    Ignoring self-intersections.

    block_sets: a list `len(num_blocks)` containing lists with each blocks'
                input point coordinates.
    circumcenters: an ndarray of double `shape(ntrias,ndim)`. Coordinates of
        the circumcenters.
    radii: an ndarray of double `shape(ntrias,1)`. Radius of circumcenters.
    rank: an int. Represents the owner rank of the block.
    """
    num_trias = len(circumcenters)
    ndim = len(circumcenters[0])

    assert num_trias > 0, "too few points"
    assert ndim > 1 or ndim < 4, "ndim is wrong"

    intersects = np.zeros((num_trias, 5), dtype=int) - 1
    num_intersects = np.zeros((num_trias, 1), dtype=int)
    for tria, (cc, rr) in enumerate(zip(circumcenters, radii)):
        for block_num, block in enumerate(block_sets):
            le = np.amin(block, axis=0)
            re = np.amax(block, axis=0)
            if __do_intersect(ndim, cc, rr, le, re) and block_num != rank:
                intersects[tria, num_intersects[tria]] = block_num
                num_intersects[tria] += 1
    return num_intersects, intersects


def __do_intersect(ndim, c, r, le, re):
    """
    Return if a sphere intersects a box
    """
    for i in range(ndim):
        if c[i] < le[i]:
            if c[i] + r < le[i]:
                return False
        elif c[i] > re[i]:
            if c[i] - r > re[i]:
                return False
    return True


def calc_circumballs(points, vertices):
    """
    Returns the balls that inscribe the triangles defined by points.

    points: an ndarray of double,`shape(npoints,ndim)`. Coordinates of the
            input points.
    vertices: an ndarray of int, `(ndarray of ints, shape (nsimplex, ndim+1)`.
            Indices of the points forming the simplices in the triangulation.
            For, 2D the points should be counterclockwise
    """
    num_points, ndim = points.shape

    assert num_points > 3, "too few points"
    assert ndim > 1 or ndim < 4, "ndim is wrong"

    p = points[vertices]

    A = p[:, 0, :].T
    B = p[:, 1, :].T
    C = p[:, 2, :].T

    num_trias = len(A.T)

    if ndim < 3:
        A = np.append(A, [np.zeros(num_trias)], axis=0)
        B = np.append(B, [np.zeros(num_trias)], axis=0)
        C = np.append(C, [np.zeros(num_trias)], axis=0)

    a = A - C
    b = B - C

    norm = np.linalg.norm
    cross = np.cross

    # https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations
    term1_a1 = norm(a.T, 2, axis=1).T ** 2
    term1_b1 = norm(b.T, 2, axis=1).T ** 2
    term1 = term1_a1[:, None] * b.T - term1_b1[:, None] * a.T
    term2 = cross(a.T, b.T, axis=1)
    term3 = 2 * norm(cross(a.T, b.T, axis=1), 2, axis=1) ** 2
    term3 = term3[:, None]
    circumcenters = np.array((cross(term1, term2, axis=1) / term3) + C.T)

    radii = np.array(
        (norm(a.T, 2, axis=1).T * norm(b.T, 2, axis=1).T * norm(a.T - b.T, 2, axis=1).T)
        / (2 * norm(cross(a.T, b.T, axis=1), 2, axis=1).T)
    )
    radii = radii[:, None]

    # delete dummy third dimension
    if ndim == 2:
        circumcenters = np.delete(circumcenters, 2, 1)
    return circumcenters, radii


def plot_circumballs(points, simplices, cc, rr):
    """
    Visualize circumcircles ontop of the triangulation
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections

    fig, ax = plt.subplots()
    plt.triplot(points[:, 0], points[:, 1], simplices.copy(), c="#FFAC67")
    patches = [plt.Circle(center, size, fill=None) for center, size in zip(cc, rr)]
    coll = matplotlib.collections.PatchCollection(patches, match_original=True,)
    ax.add_collection(coll)
    plt.show()


def on_hull(p):
    """
    Return vertices in `p` represeting the convex `hull``

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(p)
    return hull.vertices


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed.
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def are_finite(points):
    """
    Determine if a set of points are `finite`.
    A point is finite when it is
    not a member of the convex hull of the point set

    `points` should be a `NxK` coordinates of `N` points in `K` dimensions
    """
    areFinite = np.ones((len(points)), dtype=bool)
    areFinite[on_hull(points)] = False
    return areFinite
