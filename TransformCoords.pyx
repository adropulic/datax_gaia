###############################################################################
# TransformCoords.pyx
###############################################################################
# Laura Chang, Princeton University, 10-04-18
###############################################################################

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt,sin,cos,acos,asin,atan2
from libc.math cimport M_PI as pi

@cython.boundscheck(False)
@cython.wraparound(False)
def galcart2pm(double[:] ra, double[:] dec, double[:] parallax, double[:] U, double[:] V, double[:] W):
	"""
	Transforms velocities from Galactic (vx,vy,vz) to ICRS (pmra, pmdec, ddot), but uses ra and dec.
	Note this assumes J2000.0 epoch. For a different epoch, consult Eq. (32) of
	https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf.
	"""
	########################
	# Define global params #
	########################
	cdef double alpha_NGP = 192.85948*pi/180
	cdef double delta_NGP = 27.12825*pi/180
	cdef double theta = 122.932*pi/180
	cdef double k = 4.74047

	cdef np.ndarray[double, ndim=2, mode='c'] T1 = np.array([[cos(theta),sin(theta),0],[sin(theta),-cos(theta),0],[0,0,1]])
	cdef np.ndarray[double, ndim=2, mode='c'] T2 = np.array([[-sin(delta_NGP),0,cos(delta_NGP)],[0,1,0],[cos(delta_NGP),0,sin(delta_NGP)]])
	cdef np.ndarray[double, ndim=2, mode='c'] T3 = np.array([[cos(alpha_NGP),sin(alpha_NGP),0],[-sin(alpha_NGP),cos(alpha_NGP),0],[0,0,1]])

	cdef np.ndarray[double, ndim=2, mode='c'] T = np.matmul(np.matmul(T1,T2),T3)

	cdef np.ndarray[double, ndim=2, mode='c'] A1 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A2 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A = np.empty((3,3),dtype=np.float)
	
	cdef np.ndarray[double, ndim=2, mode='c'] invmat = np.empty((3,3),dtype=np.float)
	cdef int nstars = len(ra)

	cdef Py_ssize_t i

	cdef np.ndarray[double, ndim=1, mode='c'] input_ary = np.empty(3,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] ddot = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] pmra = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] pmdec = np.empty(nstars,dtype=np.float)
	
	cdef double ans0, ans1, ans2

	for i in range(nstars):
		A1 = np.array([[cos(ra[i]),-sin(ra[i]),0],[sin(ra[i]),cos(ra[i]),0],[0,0,1]])
		A2 = np.array([[cos(dec[i]),0,-sin(dec[i])],[0,1,0],[sin(dec[i]),0,cos(dec[i])]])
		A = np.matmul(A1,A2)
		
		invmat = np.linalg.inv(np.matmul(T,A))
		
		input_ary[0] = U[i]
		input_ary[1] = V[i]
		input_ary[2] = W[i]

		ans0, ans1, ans2 = np.matmul(invmat,input_ary)
		ddot[i] = ans0
		pmra[i] = ans0/parallax[i]
		pmdec[i] = ans1/parallax[i]
		
	return ddot, pmra, pmdec

@cython.boundscheck(False)
@cython.wraparound(False)
def pm2galcart(double[:] ra, double[:] dec, double[:] parallax, double[:] pmra, double[:] pmdec, double[:] ddot):
	"""
	Transforms velocities from ICRS (pmra, pmdec, ddot) to Galactic (vx,vy,vz).
	Note this assumes J2000.0 epoch. For a different epoch, consult Eq. (32) of
	https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf.
	"""
	########################
	# Define global params #
	########################
	cdef double alpha_NGP = 192.85948*pi/180
	cdef double delta_NGP = 27.12825*pi/180
	cdef double theta = 122.932*pi/180
	cdef double k = 4.74047

	cdef np.ndarray[double, ndim=2, mode='c'] T1 = np.array([[cos(theta),sin(theta),0],[sin(theta),-cos(theta),0],[0,0,1]])
	cdef np.ndarray[double, ndim=2, mode='c'] T2 = np.array([[-sin(delta_NGP),0,cos(delta_NGP)],[0,1,0],[cos(delta_NGP),0,sin(delta_NGP)]])
	cdef np.ndarray[double, ndim=2, mode='c'] T3 = np.array([[cos(alpha_NGP),sin(alpha_NGP),0],[-sin(alpha_NGP),cos(alpha_NGP),0],[0,0,1]])

	cdef np.ndarray[double, ndim=2, mode='c'] T = np.matmul(np.matmul(T1,T2),T3)

	cdef np.ndarray[double, ndim=2, mode='c'] A1 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A2 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A = np.empty((3,3),dtype=np.float)

	cdef int nstars = len(ra)

	cdef Py_ssize_t i

	cdef np.ndarray[double, ndim=1, mode='c'] input_ary = np.empty(3,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] U = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] V = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] W = np.empty(nstars,dtype=np.float)

	for i in range(nstars):
		A1 = np.array([[cos(ra[i]),-sin(ra[i]),0],[sin(ra[i]),cos(ra[i]),0],[0,0,1]])
		A2 = np.array([[cos(dec[i]),0,-sin(dec[i])],[0,1,0],[sin(dec[i]),0,cos(dec[i])]])
		A = np.matmul(A1,A2)
		input_ary[0] = ddot[i]
		input_ary[1] = k/parallax[i]*pmra[i]
		input_ary[2] = k/parallax[i]*pmdec[i]

		U[i], V[i], W[i] = np.matmul(np.matmul(T,A),input_ary)
	return U, V, W

@cython.boundscheck(False)
@cython.wraparound(False)
def ICRS2GalCart(double[:] ra, double[:] dec, double[:] parallax):
	"""
	Transforms coordinates from ICRS (ra, dec, parallax) to Galactic (x,y,z).
	Note this assumes J2000.0 epoch. For a different epoch, consult Eq. (32) of
	https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf.
	"""
	cdef int nstars = len(ra)

	cdef np.ndarray[double, ndim=1, mode='c'] cc_bl = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] cs_bl = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] s_b = np.empty(nstars,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] x = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] y = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] z = np.empty(nstars,dtype=np.float)

	cdef Py_ssize_t i

	cc_bl, cs_bl, s_b = radec2lb(ra, dec, returnlb=False)

	for i in range(nstars):
		x[i] = cc_bl[i]/parallax[i]
		y[i] = cs_bl[i]/parallax[i]
		z[i] = s_b[i]/parallax[i]

	# x = cc_bl/parallax
	# y = cs_bl/parallax
	# z = s_b/parallax

	return x, y, z

@cython.boundscheck(False)
@cython.wraparound(False)
def radec2lb(double[:] ra, double[:] dec, returnlb=True):
	"""
	Transforms coordinates from ICRS (ra, dec) to Galactic (l,b).
	Note this assumes J2000.0 epoch. For a different epoch, consult Eq. (32) of
	https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf.
	"""
	########################
	# Define global params #
	########################
	cdef double alpha_NGP = 192.85948*pi/180
	cdef double delta_NGP = 27.12825*pi/180
	cdef double theta = 122.932*pi/180

	cdef np.ndarray[double, ndim=2, mode='c'] T1 = np.array([[cos(theta),sin(theta),0],[sin(theta),-cos(theta),0],[0,0,1]])
	cdef np.ndarray[double, ndim=2, mode='c'] T2 = np.array([[-sin(delta_NGP),0,cos(delta_NGP)],[0,1,0],[cos(delta_NGP),0,sin(delta_NGP)]])
	cdef np.ndarray[double, ndim=2, mode='c'] T3 = np.array([[cos(alpha_NGP),sin(alpha_NGP),0],[-sin(alpha_NGP),cos(alpha_NGP),0],[0,0,1]])

	cdef np.ndarray[double, ndim=2, mode='c'] T = np.matmul(np.matmul(T1,T2),T3)

	#######################
	# Things to solve for #
	#######################

	cdef int nstars = len(ra)

	cdef np.ndarray[double, ndim=1, mode='c'] cc_decra = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] cs_decra = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] s_dec = np.empty(nstars,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] cc_bl = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] cs_bl = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] s_b = np.empty(nstars,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] l = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] b = np.empty(nstars,dtype=np.float)

	cdef Py_ssize_t i, j
	cdef double temp1, temp2, temp3
	cdef np.ndarray[double, ndim=1, mode='c'] temp_ary = np.empty(3,dtype=np.float)

	for i in range(nstars):
		cc_decra[i] = cos(dec[i])*cos(ra[i])
		cs_decra[i] = cos(dec[i])*sin(ra[i])
		s_dec[i] = sin(dec[i])

	for i in range(nstars):
		temp0 = 0
		temp1 = 0 
		temp2 = 0

		temp_ary[0] = cc_decra[i]
		temp_ary[1] = cs_decra[i]
		temp_ary[2] = s_dec[i]

		for j in range(3):
			temp0 += T[0,j]*temp_ary[j]
			temp1 += T[1,j]*temp_ary[j]
			temp2 += T[2,j]*temp_ary[j]

		cc_bl[i] = temp0
		cs_bl[i] = temp1
		s_b[i] = temp2

	if returnlb:
		for i in range(nstars):	
			b[i] = asin(s_b[i])
			l[i] = atan2(cs_bl[i],cc_bl[i])
		return l,b
	else:
		return cc_bl, cs_bl, s_b

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate2d(double[:] v, double theta, deg=False):
	"""
	Rotates 2d vector v (counterclockwise) by angle theta. 
	:param: deg: indicates whether the input is in degrees or radians (default)
	"""
	cdef double v0 = v[0]
	cdef double v1 = v[1]
	
	cdef np.ndarray[double, ndim=1, mode='c'] vprime = np.empty(2,dtype=np.float)
	
	if deg:
		theta = theta * pi/180
	
	vprime[0] = cos(theta)*v0-sin(theta)*v1
	vprime[1] = sin(theta)*v0+cos(theta)*v1
	
	return vprime

@cython.boundscheck(False)
@cython.wraparound(False)
def rvcart2sph(double[:] coords_cart, double[:] vels_cart, deg=True):
	"""
	Transforms phase space coordinates in Cartesian coordinates from a pair of vectors [x,y,z], [v_x,v_y,v_z]
	to the corresponding pair of vectors [r,theta,phi], [v_r,v_theta,v_phi]=[r',r*sin(phi)*theta',r*phi'] 
	in spherical coordinates, where theta is the azimuthal angle and phi is the polar angle.
	:param: deg: controls whether the output angular coordinates are in degrees or radians.
	"""
	cdef double x = coords_cart[0]
	cdef double y = coords_cart[1]
	cdef double z = coords_cart[2]

	cdef double v_x = vels_cart[0]
	cdef double v_y = vels_cart[1]
	cdef double v_z = vels_cart[2]
	
	cdef double r, theta, phi, rprime, v_r, v_theta, v_phi
	cdef np.ndarray[double, ndim=1, mode='c'] coords_sph = np.empty(3,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vels_sph = np.empty(3,dtype=np.float)
	
	r = sqrt(x**2+y**2+z**2)
	theta = atan2(y,x)
	phi = acos(z/r)
	
	if deg:
		theta = theta*180/pi
		phi = phi*180/pi
	
	coords_sph[0] = r
	coords_sph[1] = theta
	coords_sph[2] = phi
	
	rprime = sqrt(x**2+y**2)
	v_r = (x*v_x+y*v_y+z*v_z)/r
	v_theta = (x*v_y-y*v_x)/rprime
	v_phi = (z*(x*v_x+y*v_y)-rprime**2*v_z)/(r*rprime)
	
	vels_sph[0] = v_r
	vels_sph[1] = v_theta
	vels_sph[2] = v_phi
	
	return coords_sph, vels_sph

@cython.boundscheck(False)
@cython.wraparound(False)
def rvcart2sph_vec(double[:,:] coords_cart, double[:,:] vels_cart, deg=True):
	"""
	Transforms phase space coordinates in Cartesian coordinates from pairs of vectors [x,y,z], [v_x,v_y,v_z]
	to the corresponding pairs of vectors [r,theta,phi], [v_r,v_theta,v_phi]=[r',r*sin(phi)*theta',r*phi'] 
	in spherical coordinates, where theta is the azimuthal angle and phi is the polar angle.

	The dimensions of the inputs and outputs are (N,3), where N = number of stars.

	:param: deg: controls whether the output angular coordinates are in degrees or radians.
	"""
	cdef double[:] x = coords_cart[:,0]
	cdef double[:] y = coords_cart[:,1]
	cdef double[:] z = coords_cart[:,2]

	cdef double[:] v_x = vels_cart[:,0]
	cdef double[:] v_y = vels_cart[:,1]
	cdef double[:] v_z = vels_cart[:,2]
	
	cdef int N = len(x)
	cdef Py_ssize_t i
	cdef double r, theta, phi, rprime, v_r, v_theta, v_phi
	cdef np.ndarray[double, ndim=2, mode='c'] coords_sph = np.empty((N,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] vels_sph = np.empty((N,3),dtype=np.float)
	

	for i in range(N):
		r = sqrt(x[i]**2+y[i]**2+z[i]**2)
		theta = atan2(y[i],x[i])
		phi = acos(z[i]/r)
	
		if deg:
			theta = theta*180/pi
			phi = phi*180/pi
	
		coords_sph[i,0] = r
		coords_sph[i,1] = theta
		coords_sph[i,2] = phi
	
		rprime = sqrt(x[i]**2+y[i]**2)
		v_r = (x[i]*v_x[i]+y[i]*v_y[i]+z[i]*v_z[i])/r
		v_theta = (x[i]*v_y[i]-y[i]*v_x[i])/rprime
		v_phi = (z[i]*(x[i]*v_x[i]+y[i]*v_y[i])-rprime**2*v_z[i])/(r*rprime)
	
		vels_sph[i,0] = v_r
		vels_sph[i,1] = v_theta
		vels_sph[i,2] = v_phi
	
	return coords_sph, vels_sph

@cython.boundscheck(False)
@cython.wraparound(False)
def rcart2sph_vec(double[:,:] coords_cart, deg=True):
	"""
	Transforms position space coordinates in Cartesian coordinates from vectors [x,y,z]
	to the corresponding vectors [r,theta,phi] in spherical coordinates, where theta is 
	the azimuthal angle and phi is the polar angle.

	The dimensions of the inputs and outputs are (N,3), where N = number of stars.

	:param: deg: controls whether the output angular coordinates are in degrees or radians.
	"""
	cdef double[:] x = coords_cart[:,0]
	cdef double[:] y = coords_cart[:,1]
	cdef double[:] z = coords_cart[:,2]
	
	cdef int N = len(x)
	cdef Py_ssize_t i
	cdef double r, theta, phi
	cdef np.ndarray[double, ndim=2, mode='c'] coords_sph = np.empty((N,3),dtype=np.float)	

	for i in range(N):
		r = sqrt(x[i]**2+y[i]**2+z[i]**2)
		theta = atan2(y[i],x[i])
		phi = acos(z[i]/r)
	
		if deg:
			theta = theta*180/pi
			phi = phi*180/pi
	
		coords_sph[i,0] = r
		coords_sph[i,1] = theta
		coords_sph[i,2] = phi
	
	return coords_sph

@cython.boundscheck(False)
@cython.wraparound(False)
def rvcart2cyl(double[:] coords_cart, double[:] vels_cart, deg=True):
	"""
	Transforms phase space coordinates in Cartesian coordinates from a pair of vectors [x,y,z], [v_x,v_y,v_z]
	to the corresponding pair of vectors [rho,theta,z], [v_rho,v_theta,v_z]=[rho',r*theta',z'] 
	in cylindrical coordinates, where theta is the azimuthal angle.
	:param: deg: controls whether the output angular coordinates are in degrees or radians.
	"""
	cdef double x = coords_cart[0]
	cdef double y = coords_cart[1]
	cdef double z = coords_cart[2]

	cdef double v_x = vels_cart[0]
	cdef double v_y = vels_cart[1]
	cdef double v_z = vels_cart[2]
	
	cdef double rho, theta, v_rho, v_theta
	cdef np.ndarray[double, ndim=1, mode='c'] coords_cyl = np.empty(3,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vels_cyl = np.empty(3,dtype=np.float)
	
	rho = sqrt(x**2+y**2)
	theta = atan2(y,x)
	
	if deg:
		theta = theta*180/pi
	
	coords_cyl[0] = rho
	coords_cyl[1] = theta
	coords_cyl[2] = z
	
	v_rho = (x*v_x+y*v_y)/rho
	v_theta = (x*v_y-y*v_x)/rho
	
	vels_cyl[0] = v_rho
	vels_cyl[1] = v_theta
	vels_cyl[2] = v_z
	
	return coords_cyl, vels_cyl

@cython.boundscheck(False)
@cython.wraparound(False)
def rvcart2cyl_vec(double[:,:] coords_cart, double[:,:] vels_cart, deg=True):
	"""
	Transforms phase space coordinates in Cartesian coordinates from pairs of vectors [x,y,z], [v_x,v_y,v_z]
	to the corresponding pairs of vectors [rho,theta,z], [v_rho,v_theta,v_z]=[rho',r*theta',z'] 
	in cylindrical coordinates, where theta is the azimuthal angle.

	The dimensions of the inputs and outputs are (N,3), where N = number of stars.

	:param: deg: controls whether the output angular coordinates are in degrees or radians.
	"""
	cdef double[:] x = coords_cart[:,0]
	cdef double[:] y = coords_cart[:,1]
	cdef double[:] z = coords_cart[:,2]

	cdef double[:] v_x = vels_cart[:,0]
	cdef double[:] v_y = vels_cart[:,1]
	cdef double[:] v_z = vels_cart[:,2]

	cdef int N = len(x)
	cdef Py_ssize_t i	
	cdef double rho, theta, v_rho, v_theta
	cdef np.ndarray[double, ndim=2, mode='c'] coords_cyl = np.empty((N,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] vels_cyl = np.empty((N,3),dtype=np.float)
	
	for i in range(N):
		rho = sqrt(x[i]**2+y[i]**2)
		theta = atan2(y[i],x[i])
		
		if deg:
			theta = theta*180/pi
		
		coords_cyl[i,0] = rho
		coords_cyl[i,1] = theta
		coords_cyl[i,2] = z[i]
		
		v_rho = (x[i]*v_x[i]+y[i]*v_y[i])/rho
		v_theta = (x[i]*v_y[i]-y[i]*v_x[i])/rho
		
		vels_cyl[i,0] = v_rho
		vels_cyl[i,1] = v_theta
		vels_cyl[i,2] = v_z[i]
	
	return coords_cyl, vels_cyl

@cython.boundscheck(False)
@cython.wraparound(False)
def toGalcen(double[:] coords, double[:] vels, double[:] coords_LSR, double[:] vels_LSR, double phi):
	"""
	Transforms phase space coordinates in Cartesian coordinates (a pair of vectors [x,y,z], [v_x,v_y,v_z])
	from Galactic frame to Galactocentric frame following the prescription detailed in section 6.3.1 of 1806.10564.
	"""
	
	cdef double[:] xy = coords[:2]
	cdef double z = coords[2]
	
	cdef double[:] v_xy = vels[:2]
	cdef double v_z = vels[2]

	cdef np.ndarray[double, ndim=1, mode='c'] coords_gc = np.empty(3,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vels_gc = np.empty(3,dtype=np.float)

	coords_gc[:2] = rotate2d(xy,phi)
	coords_gc[2] = z
	
	vels_gc[:2] = rotate2d(v_xy,phi)
	vels_gc[2] = v_z
	
	coords_gc += coords_LSR
	vels_gc += vels_LSR
	
	return coords_gc, vels_gc

@cython.boundscheck(False)
@cython.wraparound(False)
def toGalcen_vec(double[:,:] coords, double[:,:] vels, double[:] coords_LSR, double[:] vels_LSR, double phi):
	"""
	Transforms phase space coordinates in Cartesian coordinates (a pair of vectors [x,y,z], [v_x,v_y,v_z])
	from Galactic frame to Galactocentric frame following the prescription detailed in section 6.3.1 of 1806.10564.

	The dimensions of the inputs and outputs are (N,3), where N = number of stars.
	"""
	
	cdef double[:,:] xy = coords[:,:2]
	cdef double[:] z = coords[:,2]
	
	cdef double[:,:] v_xy = vels[:,:2]
	cdef double[:] v_z = vels[:,2]

	cdef int N = len(z)
	cdef Py_ssize_t i
	cdef np.ndarray[double, ndim=2, mode='c'] coords_gc = np.empty((N,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] vels_gc = np.empty((N,3),dtype=np.float)

	for i in range(N):

		coords_gc[i,:2] = rotate2d(xy[i],phi) + coords_LSR[:2]
		coords_gc[i,2] = z[i] + coords_LSR[2]
		
		vels_gc[i,:2] = rotate2d(v_xy[i],phi) + vels_LSR[:2]
		vels_gc[i,2] = v_z[i] + vels_LSR[2]
			
	return coords_gc, vels_gc

@cython.boundscheck(False)
@cython.wraparound(False)
def error_toGalcen(double[:] ra, double[:] dec, double[:] parallax, double[:] pmra, double[:] pmdec, double[:] ddot,double[:] sigma):
	"""
	Propagates network's sigma to Galactocentric cartesian frame 
	"""
	cdef double alpha_NGP = 192.85948*pi/180
	cdef double delta_NGP = 27.12825*pi/180
	cdef double theta = 122.932*pi/180
	cdef double k = 4.74047

	cdef np.ndarray[double, ndim=2, mode='c'] T1 = np.array([[cos(theta),sin(theta),0],[sin(theta),-cos(theta),0],[0,0,1]])
	cdef np.ndarray[double, ndim=2, mode='c'] T2 = np.array([[-sin(delta_NGP),0,cos(delta_NGP)],[0,1,0],[cos(delta_NGP),0,sin(delta_NGP)]])
	cdef np.ndarray[double, ndim=2, mode='c'] T3 = np.array([[cos(alpha_NGP),sin(alpha_NGP),0],[-sin(alpha_NGP),cos(alpha_NGP),0],[0,0,1]])

	cdef np.ndarray[double, ndim=2, mode='c'] T = np.matmul(np.matmul(T1,T2),T3)

	cdef np.ndarray[double, ndim=2, mode='c'] A1 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A2 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] A = np.empty((3,3),dtype=np.float)

	cdef int nstars = len(sigma)
	cdef np.ndarray[double, ndim=2, mode='c'] COV_VRAB = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] COV_UVW = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_U = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_V = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_W = np.empty(nstars,dtype=np.float)

	for i in range(nstars):
		A1 = np.array([[cos(ra[i]),-sin(ra[i]),0],[sin(ra[i]),cos(ra[i]),0],[0,0,1]])
		A2 = np.array([[cos(dec[i]),0,-sin(dec[i])],[0,1,0],[sin(dec[i]),0,cos(dec[i])]])
		A = np.matmul(A1,A2)
		print(A)
		COV_VRAB = np.zeros_like(COV_VRAB)
		COV_VRAB[0,0] = sigma[i]*sigma[i]
		#print(np.matmul(np.matmul(T,A),np.matmul(COV_VRAB,np.matmul(np.transpose(A),np.transpose(T)))).shape)
		COV_UVW = np.matmul(np.matmul(T,A),np.matmul(COV_VRAB,np.matmul(np.transpose(A),np.transpose(T))))
		COV_U[i] = COV_UVW[0,0]
		COV_V[i] = COV_UVW[1,1]
		COV_W[i] = COV_UVW[2,2]

	return COV_U, COV_V, COV_W

@cython.boundscheck(False)
@cython.wraparound(False)
def error_toGalcen_sph(double[:] ra, double[:] dec, double[:] b, double[:] l, double[:] parallax, double[:] theta, double[:] phi, float[:] sigma, float[:] ddot, double[:] pmra, double[:] pmdec):
	"""
	Propagates network error to galactocentric spherical coordinates
	"""
	cdef double alpha_NGP = 192.85948*pi/180
	cdef double delta_NGP = 27.12825*pi/180
	cdef double k = 4.74047
	cdef np.ndarray[double, ndim=1 , mode='c'] solar_corr = np.array([11.1, 239.08, 7.25])
	cdef double galcen_distance = 8.0004 # in kpc
	cdef double z_sun = 0.015 # in kpc
	cdef double mas_per_rad = 2.062648062471e8
	cdef double s_per_year = 31557600
	cdef double km_per_kpc = 3.08567758e16        

	cdef np.ndarray[double, ndim=2, mode='c'] P = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] mat_1 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] mat_2 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] mat_3 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] mat_4 = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] M = np.empty((3,3),dtype=np.float)

	cdef int nstars = len(sigma)
	cdef np.ndarray[double, ndim=2, mode='c'] COV_VLOSAD = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=2, mode='c'] COV_VRTP = np.empty((3,3),dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_VR = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_VTHETA = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] COV_VPHI = np.empty(nstars,dtype=np.float)

	cdef np.ndarray[double, ndim=1, mode='c'] coords_sph = np.empty(3,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] coords_cart = np.empty(3,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vr  = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vtheta  = np.empty(nstars,dtype=np.float)
	cdef np.ndarray[double, ndim=1, mode='c'] vphi  = np.empty(nstars,dtype=np.float)
	cdef double fact = km_per_kpc / mas_per_rad /s_per_year

	cdef double sin_theta_sol = z_sun/galcen_distance
	cdef double cos_theta_sol = np.sqrt(1. - sin_theta_sol**2)
	cdef np.ndarray[double, ndim=2, mode='c'] mat_sol = np.array([[ cos_theta_sol, 0, sin_theta_sol],[0,1,0],[-sin_theta_sol, 0, cos_theta_sol]])

	for i in range(nstars):
		COV_VLOSAD = np.zeros_like(COV_VLOSAD)
		COV_VLOSAD[0,0] = sigma[i]*sigma[i]
		cos_phi_conv = (sin(delta_NGP) - sin(dec[i]) * sin(b[i])) / (cos(dec[i]) * cos(b[i]))
		sin_phi_conv = sin(ra[i] - alpha_NGP) * cos(delta_NGP) / cos(b[i])
		P = np.array([[1,0,0],[0,cos_phi_conv, sin_phi_conv],[0,-sin_phi_conv, cos_phi_conv]])
		mat_1 = np.array([[cos(theta[i]), 0, sin(theta[i])],[0,1,0],[-sin(theta[i]), 0, cos(theta[i])]])
		mat_2 = np.array([[cos(phi[i]), sin(phi[i]), 0],[-sin(phi[i]), cos(phi[i]), 0],[0,0,1]])
		mat_3 = np.array([[cos(l[i]), -sin(l[i]), 0], [sin(l[i]),  cos(l[i]), 0],[0,0,1]])
		mat_4 = np.array([[cos(b[i]), 0, -sin(b[i])], [0,1,0],[sin(b[i]), 0, cos(b[i])]])
        
		M = np.matmul(np.matmul(np.matmul(np.matmul(mat_1,mat_2),mat_sol),mat_3),mat_4)
		COV_VRTP = np.matmul(np.matmul(M,P),np.matmul(COV_VLOSAD,np.matmul(np.transpose(P),np.transpose(M))))
		COV_VR[i] = np.sqrt(COV_VRTP[0,0])
		COV_VTHETA[i] = np.sqrt(COV_VRTP[1,1])
		COV_VPHI[i] = np.sqrt(COV_VRTP[2,2])
		
		coords_cart = np.array([ddot[i], pmra[i]*(1/parallax[i])*cos(dec[i])*fact, pmdec[i]*(1/parallax[i])*fact])
		coords_sph = np.matmul(np.matmul(M,P), coords_cart) + np.matmul(np.matmul(mat_1, mat_2), solar_corr)
		vr[i] = coords_sph[0]
		vtheta[i] = coords_sph[1]
		vphi[i] = coords_sph[2]
		
	return COV_VR, COV_VTHETA, COV_VPHI,vr, vtheta, vphi
