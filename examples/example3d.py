import numpy as np
import simple_cgal

import time

num_points = 1000
points = np.random.random(size=(num_points, 3))

t1 = time.time()
cells = np.array(
    simple_cgal.delaunay3(points[:,0], points[:,1], points[:,2])
)
print('elapsed time is '+str(time.time()-t1))
print("Simplices shape: ", cells.shape)
print("First 10 tetrahedra: ")
print(*cells[:10], sep='\n')
