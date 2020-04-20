import time
import numpy as np
from mpi4py import MPI
from scipy.spatial import Delaunay as SciPyDelaunay
import matplotlib.pyplot as plt

import utils

# structrued grid
gpoints = [
    [0, 0],
    [0, 0.25],
    [0, 0.5],
    [0, 0.75],
    [0, 1],
    [0.25, 0],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.25, 0.75],
    [0.25, 1],
    [0.5, 0],
    [0.5, 0.25],
    [0.5, 0.5],
    [0.5, 0.75],
    [0.5, 1],
    [0.75, 0],
    [0.75, 0.25],
    [0.75, 0.5],
    [0.75, 0.75],
    [0.75, 1],
    [1, 0],
    [1, 0.25],
    [1, 0.5],
    [1, 0.75],
    [1, 1],
]
num_points = len(gpoints)
# add some noise
np.random.seed(5)
noise = np.random.uniform(size=(num_points, 2), low=-0.01, high=0.01)
gpoints += noise
# triangulation to check with
gtria = SciPyDelaunay(gpoints)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t1 = time.time()

points, extents = utils.blocker(points=gpoints, rank=rank, nblocks=size)

tria = SciPyDelaunay(points, incremental=True)

exports = utils.enqueue(extents, points, tria.vertices, rank, size)

new_points = utils.migration(comm, rank, size, exports)

tria.add_points(new_points, restart=True)

points, faces = utils.remove_external_faces(tria.points, tria.vertices, extents[rank])

upoints, ufaces = utils.aggregate(points, faces, comm, size, rank)

comm.barrier()
if rank == 0:
    print("elapsed time is " + str(time.time() - t1), flush=True)
    print(upoints.shape)
    print(ufaces.shape)
    print(ufaces[:5, :])
    fig, ax = plt.subplots()
    plt.triplot(upoints[:, 0], upoints[:, 1], ufaces)
    plt.show()

    fig, ax = plt.subplots()
    plt.triplot(gtria.points[:, 0], gtria.points[:, 1], gtria.vertices)
    print(gtria.vertices.shape)
    plt.show()
