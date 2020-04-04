import numpy as np

import simple_cgal as cgal
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


def which_intersect(block_sets, circumcenters, radii, rank):
    """
    Returns the block # each circumcircle intersects with.
    0 for block below, 1 for block above. If -1, circle does
    intersect neighboring block.

    """
    if rank == 0:
        nei_blocks = [block_sets[rank + 1]]
    elif rank == len(block_sets) - 1:
        nei_blocks = [block_sets[rank - 1]]
    else:
        nei_blocks = [block_sets[rank - 1], block_sets[rank + 1]]

    le = np.array([np.amin(block, axis=0) for block in nei_blocks]).flatten()
    re = np.array([np.amax(block, axis=0) for block in nei_blocks]).flatten()

    # add dummy box if rank==0 or rank=size-1
    if len(le) == 2:
        le = np.append(le, [-9999, -9999])
        re = np.append(re, [-9998, -9998])

    intersect = cutils.sph_bx_intersect2(circumcenters, radii, le, re)

    return intersect


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
