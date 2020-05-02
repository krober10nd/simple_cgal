import numpy as np

import simple_cgal as cgal
import cpputils as cutils


"""
Utilities for the parallel Delaunay algorithm.
"""


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements="C")
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize * ncolumns))
    B, I, J = np.unique(A.view(dtype), return_index=True, return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order="C")

    # There must be a better way to do this:
    if return_index:
        if return_inverse:
            return B, I, J
        else:
            return B, I
    else:
        if return_inverse:
            return B, J
        else:
            return B


def fixmesh(p, t, ptol=2e-13):
    """Remove duplicated/unused nodes
    Parameters
    ----------
    p : array, shape (np, dim)
    t : array, shape (nt, nf)
    Usage
    -----
    p, t = fixmesh(p, t, ptol)
    """
    snap = (p.max(0) - p.min(0)).max() * ptol
    _, ix, jx = unique_rows(np.round(p / snap) * snap, True, True)

    p = p[ix]
    t = jx[t]

    t = np.sort(t, axis=1)
    t = unique_rows(t)

    return p, t


def drectangle(p, x1, x2, y1, y2):
    min = np.minimum
    """Signed distance function for rectangle with corners (x1,y1), (x2,y1),
    (x1,y2), (x2,y2).
    This has an incorrect distance to the four corners but that isn't a big deal
    """
    return -min(min(min(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])


def remove_external_faces(points, faces, extents):
    """
    Remove faces with all three vertices outside block (external)
    """
    signed_distance = drectangle(
        points[faces.flatten(), :],
        x1=extents[0],
        x2=extents[2],
        y1=extents[1],
        y2=extents[3],
    )
    isOut = np.reshape(signed_distance > 0, (-1, 3))
    faces_new = faces[np.sum(isOut, axis=1) != 3, :]
    points_new, faces_new = fixmesh(points, faces_new)
    return points_new, faces_new


def vertex_to_elements(points, faces):
    """
    Determine which elements connected to vertices
    """
    num_faces = len(faces)

    ext = np.tile(np.arange(0, num_faces), (3, 1)).reshape(-1, order="F")
    ve = np.reshape(faces, (-1,))
    ve = np.vstack((ve, ext)).T
    ve = ve[ve[:, 0].argsort(), :]

    idx = np.insert(np.diff(ve[:, 0]), 0, 0)
    vtoe_pointer = np.argwhere(idx)
    vtoe_pointer = np.insert(vtoe_pointer, 0, 0)
    vtoe_pointer = np.append(vtoe_pointer, num_faces * 3)

    vtoe = ve[:, 1]

    return vtoe, vtoe_pointer


def calc_circumballs(points, faces):
    """
    Returns the balls that inscribe the triangles defined by points.
    Calls a pybind11 CPP subroutine in src/delaunay.cpp

    points: an ndarray of double,`shape(npoints,ndim)`. Coordinates of the
            input points.
    faces: an ndarray of int, `(ndarray of ints, shape (nsimplex, ndim+1)`.
            Indices of the points forming the simplices in the triangulation.
            For, 2D the points should be counterclockwise
    """
    num_points, ndim = points.shape

    assert num_points > 3, "too few points"
    assert ndim > 1 or ndim < 4, "ndim is wrong"

    tmp = cgal.circumballs2(points[faces, :].flatten())
    circumcenters = tmp[:, 0:2]
    radii = np.sqrt(tmp[:, 2])
    return circumcenters, radii


def plot_circumballs(points, faces, cc, rr):
    """
    Visualize circumcircles ontop of the triangulation
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections

    fig, ax = plt.subplots()
    plt.triplot(points[:, 0], points[:, 1], faces.copy(), c="#FFAC67")
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
