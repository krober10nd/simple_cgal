import time
import numpy as np
import simple_cgal
import matplotlib.pyplot as plt

num_points = 1000
points = np.random.random(size=(num_points, 2))

t1 = time.time()
faces = simple_cgal.delaunay2(points[:,0], points[:,1])
print('elapsed time is '+str(time.time()-t1))
print(faces.shape)
print(faces[:5, :])
plt.triplot(points[:,0], points[:,1], faces)
plt.show()
