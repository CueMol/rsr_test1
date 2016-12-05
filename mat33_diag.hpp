#ifndef MAT33_DIAG_HPP_INCLUDED
#define MAT33_DIAG_HPP_INCLUDED

static inline
void mat33_diag(const Matrix3D &mat, Matrix3D &evecs, Vector4D &evals)
{
  double p1 = mat.aij(1,2)*mat.aij(1,2) + mat.aij(1,3)*mat.aij(1,3) + mat.aij(2,3)*mat.aij(2,3);


  if (p1==0.0) {
    evals.x() = mat.aij(1,1);
    evals.y() = mat.aij(2,2);
    evals.z() = mat.aij(3,3);
  }
  else {
    Matrix3D I;

    double q = (mat.aij(1,1) + mat.aij(2,2) + mat.aij(3,3))/3.0;
    double p2 = (mat.aij(1,1) - q)^2 + (mat.aij(2,2) - q)^2 + (mat.aij(3,3) - q)^2 + 2.0 * p1;
    double p = sqrt(p2 / 6.0);

    //B = (1 / p) * (A - q * I); // I is the identity matrix
    Matrix3D B = (mat - I.scale(q)).scale(1.0/p);

    double r = B.deter() / 2.0;

    // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    // but computation error can leave it slightly outside this range.
    double phi;
    if (r <= -1.0) 
      phi = pi / 3.0;
    else if (r >= 1.0)
      phi = 0.0;
    else
      phi = acos(r) / 3.0;

    // the eigenvalues satisfy eig3 <= eig2 <= eig1
    evals.x() = q + 2.0 * p * cos(phi);
    evals.y() = q + 2.0 * p * cos(phi + (2.0*M_PI/3));
    evals.z() = 3.0 * q - eig1 - eig3;     // since trace(A) = eig1 + eig2 + eig3

    Matrix3D ev1 = (mat-I.scale(evals.y())).matprod(mat-I.scale(evals.z()));
    Matrix3D ev2 = (mat-I.scale(evals.z())).matprod(mat-I.scale(evals.x()));
    Matrix3D ev3 = (mat-I.scale(evals.x())).matprod(mat-I.scale(evals.y()));

    evecs.aij(1,1) = ev1.aij(1,1);
    evecs.aij(2,1) = ev1.aij(1,2);
    evecs.aij(3,1) = ev1.aij(1,3);

    evecs.aij(1,2) = ev2.aij(1,1);
    evecs.aij(2,2) = ev2.aij(1,2);
    evecs.aij(3,2) = ev2.aij(1,3);

    evecs.aij(1,3) = ev3.aij(1,1);
    evecs.aij(2,3) = ev3.aij(1,2);
    evecs.aij(3,3) = ev3.aij(1,3);
  }
}

#endif

