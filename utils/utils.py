import numpy as np

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
    num_points = int(np.amax(faces) + 1)
    nne = np.zeros((num_points), dtype=int)
    for face in faces:
        for vertex in face:
            nne[vertex] += 1
    mnz = int(np.amax(nne) + 1)
    nne.fill(0)
    vtoe = np.zeros((mnz, num_points), dtype=int)
    for ie, face in enumerate(faces):
        for vertex in face:
            vtoe[nne[vertex], vertex] = ie
            nne[vertex] += 1
    return vtoe, nne




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

    radii = []
    circumcenters = []
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations
    for pa, pb, pc in zip(a.T, b.T, C.T):
        term1 = (norm(pa, 2) ** 2) * pb - (norm(pb, 2) ** 2) * pa
        term2 = cross(pa, pb)
        term3 = 2 * norm(cross(pa, pb)) ** 2
        circumcenters.append((cross(term1, term2) / term3) + pc)

        radii.append(
            (norm(pa, 2) * norm(pb, 2) * norm(pa - pb, 2)) / (2 * norm(cross(pa, pb)))
        )

    # delete dummy third dimension
    if ndim == 2:
        for ix, row in enumerate(circumcenters):
            circumcenters[ix] = np.delete(row, 2)

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


def are_finite(p):
    """
    Determine if a set of points are `finite`.
    A point is finite when it is
    not a member of the convex hull of the point set

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    """
    areFinite = np.ones((len(p)), dtype=bool)
    areFinite[on_hull(p)] = False
    return areFinite
