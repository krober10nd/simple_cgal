import numpy as np
import utils
import cpputils as cutils

"""
Migration routines for moving points during parallel Delaunay
"""


def enqueue(extents, points, faces, rank, size):
    """
    Return ranks that cell sites (vertices of triangulation) need to be sent
    """
    # determine llc (lower left corner) and urc (upper right corner)
    if rank == 0:
        le = [extents[rank + 1][0:2]]
        re = [extents[rank + 1][2:4]]
    elif rank == size - 1:
        le = [extents[rank - 1][0:2]]
        re = [extents[rank - 1][2:4]]
    else:
        le = [extents[rank - 1][0:2], extents[rank + 1][0:2]]
        re = [extents[rank - 1][2:4], extents[rank + 1][2:4]]

    # add dummy box above if rank==0 or under if rank=size-1
    if rank == size - 1:
        le = np.append(le, [-9999, -9999])
        re = np.append(re, [-9998, -9998])

    if rank == 0:
        le = np.insert(le, 0, [-9999, -9999])
        re = np.insert(re, 0, [-9998, -9998])

    exports = cutils.where_to2(points, faces, le, re, rank)

    return exports


def migration(comm, rank, size, exports):  # points):
    """
    Transmit data via MPI using P2P comm
    """
    NSB = int(exports[0, 0])
    NSA = int(exports[0, 1])

    tmp = []
    if NSB != 0:
        comm.send(exports[1 : NSB + 1, :], dest=rank - 1, tag=11)
        tmp = np.append(tmp, comm.recv(source=rank - 1, tag=11))

    if NSA != 0:
        comm.send(exports[NSB + 1 : NSB + 1 + NSA, :], dest=rank + 1, tag=11)
        tmp = np.append(tmp, comm.recv(source=rank + 1, tag=11))

    new_points = np.reshape(tmp, (int(len(tmp) / 2), 2))
    return new_points
