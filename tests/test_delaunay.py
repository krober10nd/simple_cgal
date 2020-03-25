import numpy as np
import simple_cgal


def test_one_triangle():
    # * - *
    #  \ /
    #   *
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 0.5, 1.0])
    faces = simple_cgal.delaunay2(x, y)
    assert faces.shape == (1, 3)
    assert set(faces[0]) == {0, 1, 2}


def test_two_triangles():

    #   *
    #  / \
    # * - *
    #  \ /
    #   *
    x = np.array([0.0, 0.5, 1.0, 0.5])
    y = np.array([1.0, 0.3, 1.0, 1.7])
    faces = simple_cgal.delaunay2(x, y)
    assert faces.shape == (2, 3)
    f_sets = [set(f) for f in faces]
    assert  {0, 1, 2} in f_sets
    assert  {0, 2, 3} in f_sets


def test_2D_random():
    points = np.random.random(size=(100, 2))
    faces = np.array(
        simple_cgal.delaunay2(points[:, 0], points[:, 1])
    )
    assert faces.min() == 0
    assert faces.max() == 99


def test_3D_random():
    points = np.random.random(size=(100, 3))
    cells = np.array(
        simple_cgal.delaunay3(points[:, 0], points[:, 1], points[:, 2])
    )
    assert cells.min() == 0
    assert cells.max() == 99


def test_centered_cube():

    x = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    z = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5])

    cells = simple_cgal.delaunay3(x, y, z)
    assert cells.shape == (12, 4)
