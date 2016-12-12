#ifndef MAT33_DIAG_HPP_INCLUDED
#define MAT33_DIAG_HPP_INCLUDED

namespace {
  using qlib::Vector4D;
  using qlib::Matrix3D;

  template <typename T>
  T sqr(T val) { return val*val; }
  
  inline
  void mkevec(const Matrix3D &mat, const Vector4D &ev, Matrix3D &evecs, int i, int j, int k)
  {
    Vector4D v1;
    v1.x() = (mat.aij(1,1)-ev.ai(j))*(mat.aij(1,1)-ev.ai(k)) + mat.aij(1,2)*mat.aij(2,1) + mat.aij(1,3)*mat.aij(3,1);
    v1.y() = (mat.aij(1,1)-ev.ai(j))*mat.aij(1,2) + mat.aij(1,2)*(mat.aij(2,2)-ev.ai(k)) + mat.aij(1,3)*mat.aij(3,2);
    v1.z() = (mat.aij(1,1)-ev.ai(j))*mat.aij(1,3) + mat.aij(1,2)*mat.aij(2,3) + mat.aij(1,3)*(mat.aij(3,3)-ev.ai(k));
    
    double len1 = sqrt( sqr(v1.x()) + sqr(v1.y()) + sqr(v1.z()));
    evecs.aij(1,i) = v1.x() / len1;
    evecs.aij(2,i) = v1.y() / len1;
    evecs.aij(3,i) = v1.z() / len1;
  }

  inline
  void mat33_diag(const Matrix3D &mat, Matrix3D &evecs, Vector4D &evals)
  {
      double p1 = sqr( mat.aij(1,2) ) + sqr( mat.aij(1,3) ) + sqr( mat.aij(2,3) );


      if (p1==0.0) {
        evals.x() = mat.aij(1,1);
        evals.y() = mat.aij(2,2);
        evals.z() = mat.aij(3,3);
      }
      else {
        Matrix3D I;

        double q = (mat.aij(1,1) + mat.aij(2,2) + mat.aij(3,3))/3.0;
        double p2 = sqr(mat.aij(1,1) - q) + sqr(mat.aij(2,2) - q) + sqr(mat.aij(3,3) - q) + 2.0 * p1;
        double p = sqrt(p2 / 6.0);

        //B = (1 / p) * (A - q * I); // I is the identity matrix
        Matrix3D B = (mat - I.scale(q)).scale(1.0/p);

        double r = B.deter() / 2.0;

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        double phi;
        if (r <= -1.0)
          phi = M_PI / 3.0;
        else if (r >= 1.0)
          phi = 0.0;
        else
          phi = acos(r) / 3.0;

        // the eigenvalues satisfy eig3 <= eig2 <= eig1
        evals.x() = q + 2.0 * p * cos(phi);
        evals.y() = q + 2.0 * p * cos(phi + (2.0*M_PI/3));
        evals.z() = 3.0 * q - evals.x() - evals.y();     // since trace(A) = eig1 + eig2 + eig3

        //printf("evals.x: %f\n", evals.x());
        //printf("evals.y: %f\n", evals.y());
        //printf("evals.z: %f\n", evals.z());

	/*
        Matrix3D ev1 = (mat-I.scale(evals.y())).mul(mat-I.scale(evals.z()));
        Matrix3D ev2 = (mat-I.scale(evals.z())).mul(mat-I.scale(evals.x()));
        Matrix3D ev3 = (mat-I.scale(evals.x())).mul(mat-I.scale(evals.y()));
	
        double len1 = sqrt( sqr(ev1.aij(1,1)) + sqr(ev1.aij(1,2)) + sqr(ev1.aij(1,3)));
        evecs.aij(1,2) = ev1.aij(1,1) / len1;
        evecs.aij(2,2) = ev1.aij(1,2) / len1;
        evecs.aij(3,2) = ev1.aij(1,3) / len1;

        double len2 = sqrt( sqr(ev2.aij(1,1)) + sqr(ev2.aij(1,2)) + sqr(ev2.aij(1,3)));
        evecs.aij(1,2) = ev2.aij(1,1) / len2;
        evecs.aij(2,2) = ev2.aij(1,2) / len2;
        evecs.aij(3,2) = ev2.aij(1,3) / len2;

        double len3 = sqrt( sqr(ev3.aij(1,1)) + sqr(ev3.aij(1,2)) + sqr(ev3.aij(1,3)));
        evecs.aij(1,3) = ev3.aij(1,1) / len3;
        evecs.aij(2,3) = ev3.aij(1,2) / len3;
        evecs.aij(3,3) = ev3.aij(1,3) / len3;
	*/

	//mkevec(mat, evals, evecs, 1, 2, 3);
	//mkevec(mat, evals, evecs, 2, 3, 1);
	//mkevec(mat, evals, evecs, 3, 1, 2);

	if (evals.x()<=evals.y() &&
	    evals.x()<=evals.z())
	  mkevec(mat, evals, evecs, 1, 2, 3);
	else if (evals.y()<=evals.x() &&
		 evals.y()<=evals.z())
	  mkevec(mat, evals, evecs, 1, 3, 1);
	else
	  mkevec(mat, evals, evecs, 1, 1, 2);
      }
    }
}
#endif

