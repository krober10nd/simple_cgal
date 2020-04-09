import numpy as np

import simple_cgal as cgal
import cpputils as cutils


"""
Utilities for the parallel Delaunay algorithm.
"""


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
        points, x1=extents[0], x2=extents[2], y1=extents[1], y2=extents[3]
    )

    face_new = []
    for face in faces:
        k = 0
        for vertex in face:
            d = signed_distance[vertex]
            if d > 0:
                k += 1
        if k != 3:
            face_new = np.append(face_new, face)

    face_new = np.reshape(face_new, (int(len(face_new) / 3), 3))
    return face_new


def vertex_to_elements(faces):
    """
    Returns the elements incident to a vertex in the
    Delaunay graph. Calls a pybind11 CPP subroutine in src/cpputils.cpp

    faces: an ndarray of int, `(ndarray of ints, shape (nsimplex, ndim+1)`.
            Indices of the points forming the simplices in the triangulation.
    """
    num_points = np.amax(faces) + 1
    num_faces = len(faces)
    vtoe = cutils.vertex_to_elements(faces, num_points, num_faces)
    nne = np.count_nonzero(vtoe > -1, axis=1)
    return vtoe, nne


def calc_circumballs(points, faces):
    """
    Returns the balls that inscribe the triangles defined by points.

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
