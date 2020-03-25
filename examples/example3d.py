import random

import numpy
import simple_cgal

import time

num_points = 1000
points = numpy.array([(random.random()*1.0, random.random()*1.0, random.random()*1.0) for _ in range(num_points)])

t1 = time.time()
cells = simple_cgal.delaunay3(points[:,0],points[:,1],points[:,2])
print('elapsed time is '+str(time.time()-t1))


for i in range(len(cells[0])):
    print(cells[0][i],cells[1][i],cells[2][i],cells[3][i])
print(cells.shape)
