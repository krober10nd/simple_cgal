import numpy as np

"""
Basic utilities for parallel Delaunay algorithm.
"""


def intersect_sph_box(ndim, c, r, le, re):
    """
    return if a sphere intersects a box
    """
    for i in range(ndim):
        if c[i] < le[i]:
            if c[i] + r < le[i]:
                return False
        elif c[i] > re[i]:
            if c[i] - r > re[i]:
                return False
    return True


def __dot2(u, v):
    return u[0] * v[0] + u[1] * v[1]


def __cross2(u, v, w):
    """u x (v x w)"""
    return __dot2(u, w) * v - __dot2(u, v) * w


def __ncross2(u, v):
    """|| u x v ||^2"""
    return __sq2(u) * __sq2(v) - __dot2(u, v) ** 2


def __sq2(u):
    return __dot2(u, u)


def circumballs(points, tri):
    """
    Compute the balls that inscribe the triangles
    tri must be a instance of a Delaunay triangulation
    from Scipy.Spatial.Delaunay.
    """

    p = tri.points[tri.vertices]

    # Triangle vertices
    A = p[:, 0, :].T
    B = p[:, 1, :].T
    C = p[:, 2, :].T

    a = A - C
    b = B - C

    return __cross2(__sq2(a) * b - __sq2(b) * a, a, b) / (2 * __ncross2(a, b)) + C


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


def is_finite(p):
    """
    A point/site is `finite` when it is not a member of the convex hull
    of the point set
    """
    isFinite = np.ones((len(p)), dtype=bool)
    isFinite[on_hull(p)] = False
    return isFinite
