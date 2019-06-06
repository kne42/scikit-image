cdef unsigned char point_in_polygon(Py_ssize_t nr_verts, double *xp, double *yp,
                                    double x, double y, double epsilon=*) nogil

cdef void points_in_polygon(Py_ssize_t nr_verts, double *xp, double *yp,
                            Py_ssize_t nr_points, double *x, double *y,
                            unsigned char *result, double epsilon=*) nogil
