import numpy as np
import math

def is_power_of_two(n):
    """Return True if n is a power of two."""
    if n <= 0:
        return False
    else:
        return n & (n - 1) == 0

def blocker(points, nblocks):
    ''' Decompose point coordinates into # of blocks (powers of 2) roughly balanced in # of points'''
    num_points,dim = points.shape

    assert is_power_of_two(nblocks), 'nblocks must be power of 2'
    assert nblocks >= 2, 'too few nblocks, must be >= 2'
    assert dim > 2 or dim < 3, 'dimensions of points are wrong'
    assert num_points//nblocks > 1, 'too few points for chosen nblocks'

    num_bisects = int(math.log2(nblocks))

    block_sets = __bisect(points)
    if num_bisects == 0 : return block_sets
    num_bisects -= 1
    for _ in range(num_bisects):
        store = []
        for block in block_sets:
            tmp = __bisect(np.asarray(block))
            store.append(tmp[0])
            store.append(tmp[1])
        block_sets = store
    return block_sets


def __bisect(points):
    ''' Bisect point coordinates into two half spaces'''
    num_points,dim = points.shape

    x_sorted = np.argsort(points[:, 0])
    y_sorted = np.argsort(points[:, 1])

    step = num_points//2 + 1
    ixx, iyy = np.meshgrid(
        np.arange(num_points, step=step),
            np.arange(num_points, step=step)
    )

    blocks_set = []
    for idx, idy in zip(ixx.ravel(), iyy.ravel()):
        common = set(x_sorted[idx: idx+step]).intersection(
            y_sorted[idy: idy+step]
        )
        blocks_set.append(points.take(list(common), axis=0))
    temp = []
    for i in range(0,len(blocks_set),2):
        temp.append(np.concatenate((blocks_set[i],blocks_set[i+1]),axis=0))
    blocks_set = temp
    return blocks_set
