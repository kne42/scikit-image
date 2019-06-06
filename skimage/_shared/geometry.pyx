#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.math cimport fabs


cdef enum:
    OUTSIDE = 0
    INSIDE  = 1
    VERTEX  = 2
    EDGE    = 3


cdef unsigned char point_in_polygon(Py_ssize_t nr_verts, double *xp,
                                    double *yp, double x, double y,
                                    double epsilon=0.0000001) nogil:
    """Classify where a point lies with respect to a polygon
    using the ray-crossing algorithm [1]_.

    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : double array
        Coordinates of polygon with length nr_verts.
    x, y : double
        Coordinates of point.
    epsilon : double, optional
        Error limit with which to check if point is a vertex.

    Returns
    -------
    classification : unsigned char
        Classification of the point: 0 if outside, 1 if inside, 2 if a vertex,
        and 3 if an edge.

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7
    """
    cdef:
        unsigned int r_cross = 0     # number of right edge/ray crossings
        unsigned int l_cross = 0     # number of left edge/ray crossings

        Py_ssize_t i                 # point index
        Py_ssize_t j = nr_verts - 1  # previous point index

        double x0, y0                # vertex when point is origin
        double x1, y1                # previous vertex when point is origin
    
    for i in range(nr_verts):
        x0 = xp[i] - x
        y0 = yp[i] - y

        if fabs(x0) < epsilon and fabs(y0) < epsilon:
            # it's a vertex
            return VERTEX

        x1 = xp[j] - x
        y1 = yp[j] - y

        intersection = (x0 * y1 - x1 * y0) / (y1 - y0)
        
        if (
            # straddles the x-component of the ray
            (y0 > 0) != (y1 > 0)
            # crosses the ray if strictly positive intersection
            and intersection  > 0
        ):
            r_cross += 1
            
        if (
            # straddles the x-component of the ray when reversed
            (y0 < 0) != (y1 < 0)
            # crosses the ray if strictly negative intersection
            and intersection < 0
        ):
            l_cross += 1
        
        j = i

    if r_cross % 2 != l_cross % 2:
        # on edge if left and right crossings not of same parity
        return EDGE
    elif r_cross % 2:
        # inside if odd number of right crossings
        return INSIDE
    else:
        # outside if even number of right crossings
        return OUTSIDE


cdef void points_in_polygon(Py_ssize_t nr_verts, double *xp, double *yp,
                            Py_ssize_t nr_points, double *x, double *y,
                            unsigned char *result,
                            double epsilon=0.0000001) nogil:
    """Classify where each point lies with respect to a polygon
    using the ray-crossing algorithm [1]_.


    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : double array
        Coordinates of polygon with length nr_verts.
    nr_points : int
        Number of points to test.
    x, y : double array
        Coordinates of points.
    classifications : unsigned char array
        Classifications of each point: 0 if outside, 1 if inside, 2 if a vertex,
        and 3 if an edge.
    epsilon : double, optional
        Error limit with which to check if point is a vertex.

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7
    """
    cdef Py_ssize_t n
    for n in range(nr_points):
        result[n] = point_in_polygon(nr_verts, xp, yp, x[n], y[n], epsilon)
