import random

import numpy
import simple_cgal

import time

num_points = 1000
points = numpy.array([(random.random()*1.0, random.random()*1.0) for _ in range(num_points)])

t1 = time.time()
faces = simple_cgal.delaunay2(points[:,0],points[:,1])
print('elapsed time is '+str(time.time()-t1))

import matplotlib.pyplot as plt
faces = numpy.asarray(faces).T
plt.triplot(points[:,0], points[:,1], faces)
plt.show()
