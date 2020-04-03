import numpy as np

import cpputils as cutils


"""
Utilities for the parallel Delaunay algorithm.
"""


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
    nne = np.count_nonzero(vtoe, axis=1)
    return vtoe, nne


def which_intersect(block_sets, circumballs, radii, rank):
    """

    """
    if rank == 0:
        nei_blocks = [[-9999, -9999], block_sets[rank + 1]]
    elif rank == len(block_sets) - 1:
        nei_blocks = [block_sets[rank - 1], [-9999, -9999]]
    else:
        nei_blocks = [block_sets[rank - 1], block_sets[rank + 1]]

    le = [np.amin(block, axis=0) for block in nei_blocks]
    re = [np.amax(block, axis=0) for block in nei_blocks]

    intersect, nIntersect = cutils.sph_bx_intersect(circumballs, radii, le, re)

    return nIntersect, intersect


def calc_circumballs(points, vertices):
    """
    # TODO: convert to CPP
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
