import numpy as np
from numpy.linalg import norm

"""
Utilities for the parallel Delaunay algorithm.
"""


def plot_circumballs(points, simplices, cc, rr):
    """
    Visualize circumcircles ontop of the triangulation
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections

    fig, ax = plt.subplots()
    plt.triplot(points[:, 0], points[:, 1], simplices.copy(),c='#FFAC67')
    patches = [plt.Circle(center, size, fill=None) for center, size in zip(cc, rr)]
    coll = matplotlib.collections.PatchCollection(patches, match_original=True,)
    ax.add_collection(coll)
    plt.show()


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


def circumballs(points, vertices):
    """
    Returns the balls that inscribe the triangles defined by points.

    points: an ndarray of double,`shape(npoints,ndim)`. Coordinates of the input points.
    vertices: an ndarray of int, `(ndarray of ints, shape (nsimplex, ndim+1)`. Indices of the points forming the simplices in the triangulation. For, 2D the points should be counterclockwise
    """
    num_points, ndim = points.shape

    assert num_points > 3, "too few points"
    assert ndim >= 2 or ndim < 4, "ndim is wrong"

    p = points[vertices]

    A = p[:, 0, :].T
    B = p[:, 1, :].T
    C = p[:, 2, :].T

    num_trias = len(A.T)

    A = np.append(A, [np.zeros(num_trias)], axis=0)
    B = np.append(B, [np.zeros(num_trias)], axis=0)
    C = np.append(C, [np.zeros(num_trias)], axis=0)

    a = A - C
    b = B - C

    radii = []
    circumcenters = []
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Circumcircle_equations
    for pa, pb, pc in zip(a.T, b.T, C.T):
        term1 = (norm(pa, 2) ** 2) * pb - (norm(pb, 2) ** 2) * pa
        term2 = np.cross(pa, pb)
        term3 = 2 * norm(np.cross(pa, pb)) ** 2
        circumcenters.append((np.cross(term1, term2) / term3) + pc)

        radii.append(
            (norm(pa, 2) * norm(pb, 2) * norm(pa - pb, 2))
            / (2 * norm(np.cross(pa, pb)))
        )

    return circumcenters, radii


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
    Determine if a point/site is `finite`. A point is finite when it is
    not a member of the convex hull of the point set

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    """
    isFinite = np.ones((len(p)), dtype=bool)
    isFinite[on_hull(p)] = False
    return isFinite
