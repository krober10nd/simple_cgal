# simple_cgal

# Installation
1. clone the repo: ```git clone https://github.com/krober10nd/simple_cgal.git```
2. pull the submodules: ```git submodule update --init --recursive```
3. run: ```pip install --user .```

# Requirements 
1. CGAL>=5.0 
2. Boost
3. numpy>=1.0
4. SciPy>=1.4.1
5. mpi4py
6. pybind11 
7. git
8. cmake>=2.8

# How does it work?

In serial....
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
 plt.triplot(points[:,0], points[:,1], faces)
 plt.show()
```

For parallel see `examples/example2d_parallel.py`

