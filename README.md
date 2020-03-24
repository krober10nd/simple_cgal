# simple_cgal
A simple wrapper to perform 2D/3D Delaunay triangulation using CGAL with pybind11 and CMake

# Installation
1. clone the repo: ```git clone https://github.com/krober10nd/simple_cgal.git```
2. pull the submodules: ```git submodule update --init --recursive```
3. run: ```pip install simple_cgal```

# Requirements 
CGAL>=5.0 
cmake>=2.0
numpy>=1.0

# How does it work?

```python
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
```

