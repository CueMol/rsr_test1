#ifndef CALC_DIHE_HPP_INCLUDED
#define CALC_DIHE_HPP_INCLUDED

using qlib::Vector4D;

static inline realnum_t calcAngleDiff(realnum_t phi, realnum_t phi0)
{
  realnum_t base = M_PI-phi0;
  phi += base;
  realnum_t mod = fmod(phi+4.0*M_PI, 2.0*M_PI);
  // mod: 0-360

  mod -= base;
  // mod: -base - 360-base
  //      -M_PI+phi0 - M_PI+phi0

  return mod-phi0;
}

static inline
realnum_t calcDihe(int ai, int aj, int ak, int al, const std::vector<float> &crds)
{
  Vector4D r1, r2, r3, r4;
  Vector4D r12, r23, r34, A, B, C;

  r1.x() = crds[ai+0];
  r1.y() = crds[ai+1];
  r1.z() = crds[ai+2];
  r2.x() = crds[aj+0];
  r2.y() = crds[aj+1];
  r2.z() = crds[aj+2];
  r3.x() = crds[ak+0];
  r3.y() = crds[ak+1];
  r3.z() = crds[ak+2];
  r4.x() = crds[al+0];
  r4.y() = crds[al+1];
  r4.z() = crds[al+2];

  r12 = r1 - r2;
  r23 = r2 - r3;
  r34 = r3 - r4;
    
  //  Calculate the cross products and distances
  A = r12.cross(r23);
  B = r23.cross(r34);
  C = r23.cross(A);
  
  realnum_t rA = A.length();
  realnum_t rB = B.length();
  realnum_t rC = C.length();
  
  if (qlib::isNear4<realnum_t>(rA, 0.0)) {
    printf("length of A is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }
  if (qlib::isNear4<realnum_t>(rB, 0.0)) {
    printf("length of B is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }
  if (qlib::isNear4<realnum_t>(rC, 0.0)) {
    printf("length of C is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }

  //  Calculate the sin and cos
  realnum_t cos_phi = (A.dot(B))/(rA*rB);
  realnum_t sin_phi = (C.dot(B))/(rC*rB);
  
  realnum_t phi= -::atan2f(sin_phi,cos_phi);
  
  return phi;
}

static inline
realnum_t calcDiheDiff(int ai, int aj, int ak, int al, const std::vector<float> &crds,
		       Vector4D &f1, Vector4D &f2, Vector4D &f3)
{

  Vector4D r1, r2, r3, r4;
  Vector4D r12, r23, r34, A, B, C;
  Vector4D eA, eB, eC, dcosdA, dcosdB, dsindB, dsindC;

  r1.x() = crds[ai+0];
  r1.y() = crds[ai+1];
  r1.z() = crds[ai+2];
  r2.x() = crds[aj+0];
  r2.y() = crds[aj+1];
  r2.z() = crds[aj+2];
  r3.x() = crds[ak+0];
  r3.y() = crds[ak+1];
  r3.z() = crds[ak+2];
  r4.x() = crds[al+0];
  r4.y() = crds[al+1];
  r4.z() = crds[al+2];

  r12 = r1 - r2;
  r23 = r2 - r3;
  r34 = r3 - r4;
    
  //  Calculate the cross products and distances
  A = r12.cross(r23);
  B = r23.cross(r34);
  C = r23.cross(A);
  
  realnum_t rA = A.length();
  realnum_t rB = B.length();
  realnum_t rC = C.length();
  
  if (qlib::isNear4<realnum_t>(rA, 0.0)) {
    printf("length of A is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }
  if (qlib::isNear4<realnum_t>(rB, 0.0)) {
    printf("length of B is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }
  if (qlib::isNear4<realnum_t>(rC, 0.0)) {
    printf("length of C is too small for dihe %d %d %d %d: %f\n",
	   ai/3, aj/3, ak/3, al/3, rA);
    return 0.0;
  }

  //  Calculate the sin and cos
  realnum_t cos_phi = (A.dot(B))/(rA*rB);
  realnum_t sin_phi = (C.dot(B))/(rC*rB);
  
  realnum_t phi= -::atan2f(sin_phi,cos_phi);

  //  Normalize B
  rB = 1.0/rB;
  eB = B.scale(rB); //  B *= rB;
  
  if (fabs(sin_phi) > 0.1) {
    //  use the sin version to avoid 1/cos terms
      
    //  Normalize A
    rA = 1.0/rA;
    eA = A.scale(rA);

    dcosdA = (eA.scale(cos_phi)-eB).scale(rA);
    dcosdB = (eB.scale(cos_phi)-eA).scale(rB);
      
    realnum_t K1s = -1.0/sin_phi;
      
    f1 = (r23.cross(dcosdA)).scale(K1s);
    f3 = (dcosdB.cross(r23)).scale(K1s);
    f2 = (dcosdA.cross(r12) + r34.cross(dcosdB)).scale(K1s);

  }
  else {
    //  This angle is closer to 0 or 180 than it is to
    //  90, so use the cos version to avoid 1/sin terms
      
    //  Normalize C
    rC = 1.0/rC;
    eC = C.scale(rC);
    dsindC = (eC.scale(sin_phi)-eB).scale(rC);
    dsindB = (eB.scale(sin_phi)-eC).scale(rB);
      
    realnum_t K1c = 1.0/cos_phi;
      
    f1.x() = K1c*((r23.y()*r23.y() + r23.z()*r23.z())*dsindC.x()
		  - r23.x()*r23.y()*dsindC.y()
		  - r23.x()*r23.z()*dsindC.z());
    f1.y() = K1c*((r23.z()*r23.z() + r23.x()*r23.x())*dsindC.y()
		  - r23.y()*r23.z()*dsindC.z()
		  - r23.y()*r23.x()*dsindC.x());
    f1.z() = K1c*((r23.x()*r23.x() + r23.y()*r23.y())*dsindC.z()
		  - r23.z()*r23.x()*dsindC.x()
		  - r23.z()*r23.y()*dsindC.y());
      
    // f3 = cross(K1c,dsindB,r23);
    f3 = (dsindB.cross(r23)).scale(K1c);
      
    f2.x() = K1c*(-(r23.y()*r12.y() + r23.z()*r12.z())*dsindC.x()
		  +(2.0*r23.x()*r12.y() - r12.x()*r23.y())*dsindC.y()
		  +(2.0*r23.x()*r12.z() - r12.x()*r23.z())*dsindC.z()
		  +dsindB.z()*r34.y() - dsindB.y()*r34.z());
    f2.y() = K1c*(-(r23.z()*r12.z() + r23.x()*r12.x())*dsindC.y()
		  +(2.0*r23.y()*r12.z() - r12.y()*r23.z())*dsindC.z()
		  +(2.0*r23.y()*r12.x() - r12.y()*r23.x())*dsindC.x()
		  +dsindB.x()*r34.z() - dsindB.z()*r34.x());
    f2.z() = K1c*(-(r23.x()*r12.x() + r23.y()*r12.y())*dsindC.z()
		  +(2.0*r23.z()*r12.x() - r12.z()*r23.x())*dsindC.x()
		  +(2.0*r23.z()*r12.y() - r12.z()*r23.y())*dsindC.y()
		  +dsindB.y()*r34.x() - dsindB.x()*r34.y());
  }

#ifdef DEBUG_PRINT
  printf("eA.f1=%f\n", A.normalize().dot(f1.normalize()));
  printf("eB.f3=%f\n", B.normalize().dot(f3.normalize()));
#endif

  return phi;
}

static inline
realnum_t calcDihe2(int ai, int aj, int ak, int al, const std::vector<float> &crds)
{
  Vector4D ri, rj, rk, rl;
  Vector4D rij, rkj, rkl, rmj, rnk, C;
  Vector4D rjk, A, B;

  ri.x() = crds[ai+0];
  ri.y() = crds[ai+1];
  ri.z() = crds[ai+2];
  //ri.w() = 0.0;
  rj.x() = crds[aj+0];
  rj.y() = crds[aj+1];
  rj.z() = crds[aj+2];
  //rj.w() = 0.0;
  rk.x() = crds[ak+0];
  rk.y() = crds[ak+1];
  rk.z() = crds[ak+2];
  //rk.w() = 0.0;
  rl.x() = crds[al+0];
  rl.y() = crds[al+1];
  rl.z() = crds[al+2];
  //rl.w() = 0.0;

  rij = ri - rj;
  rkj = rk - rj;
  rkl = rk - rl;
    
  //  Calculate the cross products and distances
  rmj = rij.cross(rkj);
  rnk = rkj.cross(rkl);
  
  //C = r23.cross(r12.cross(r23));
  //C = rjk.cross(rij.cross(rjk));
  C = rkj.cross(rmj);

  realnum_t lrmj = rmj.length();
  realnum_t lrnk = rnk.length();
  realnum_t lC = C.length();
  
  if (qlib::isNear4<realnum_t>(lrmj, 0.0)) {
    printf("length of Rmj is too small for dihe %d %d %d %d: %f\n", ai/3, aj/3, ak/3, al/3, lrmj);
  }
  if (qlib::isNear4<realnum_t>(lrnk, 0.0)) {
    printf("length of Rnk is too small for dihe %d %d %d %d: %f\n", ai/3, aj/3, ak/3, al/3, lrnk);
  }
  
  //  Calculate the cos
  realnum_t cos_phi = (rmj.dot(rnk))/(lrmj*lrnk);
  
  // Calculate the sign of phi
  realnum_t sign_phi = rkj.dot(rmj.cross(rnk));
  
  //realnum_t phi= copysign(1.0, sign_phi) * acos(cos_phi);
  realnum_t phi= copysign(1.0, sign_phi) * acos(qlib::trunc(cos_phi, -1.0, 1.0));

  if (!qlib::isFinite(phi)) {
    printf("phi invalid: %f sign = %f, cos_phi = %f\n", phi, sign_phi, cos_phi);
    printf("  acos(cos_phi) = %f\n", qlib::toDegree(acos(cos_phi)));
  }
  
  // //  Calculate the sin
  // realnum_t sin_phi = (C.dot(-rnk))/(lC*lrnk);
  // realnum_t phi2= -::atan2f(sin_phi,cos_phi);
  
  return phi;
}

static inline
realnum_t calcDiheDiff2(int ai, int aj, int ak, int al,
			const std::vector<float> &crds,
			Vector4D &dPhidRi, Vector4D &dPhidRl,
			Vector4D &dPhidRj, Vector4D &dPhidRk)
{
  Vector4D ri, rj, rk, rl;
  Vector4D rij, rkj, rkl, rmj, rnk;

  ri.x() = crds[ai+0];
  ri.y() = crds[ai+1];
  ri.z() = crds[ai+2];
  rj.x() = crds[aj+0];
  rj.y() = crds[aj+1];
  rj.z() = crds[aj+2];
  rk.x() = crds[ak+0];
  rk.y() = crds[ak+1];
  rk.z() = crds[ak+2];
  rl.x() = crds[al+0];
  rl.y() = crds[al+1];
  rl.z() = crds[al+2];
  
  rij = ri - rj;
  rkj = rk - rj;
  rkl = rk - rl;
    
  //  Calculate the cross products and distances
  rmj = rij.cross(rkj);
  rnk = rkj.cross(rkl);
  
  realnum_t lrmj = rmj.length();
  realnum_t lrnk = rnk.length();
  
  if (qlib::isNear4<realnum_t>(lrmj, 0.0)) {
    printf("length of Rmj is too small for dihe %d %d %d %d: %f\n", ai/3, aj/3, ak/3, al/3, lrmj);
    return 0.0;
  }
  if (qlib::isNear4<realnum_t>(lrnk, 0.0)) {
    printf("length of Rnk is too small for dihe %d %d %d %d: %f\n", ai/3, aj/3, ak/3, al/3, lrnk);
    return 0.0;
  }

  //  Calculate the cos
  realnum_t cos_phi = (rmj.dot(rnk))/(lrmj*lrnk);
  //realnum_t sin_phi = (C.dot(B))/(rC*rB);
  
  // Calculate the sign of phi
  realnum_t sign_phi = rkj.dot(rmj.cross(rnk));
  realnum_t phi= copysign(1.0, sign_phi) * acos(qlib::trunc(cos_phi, -1.0, 1.0));

  if (!qlib::isFinite(phi)) {
    printf("phi invalid: %f sign = %f, cos_phi = %f\n", phi, sign_phi, cos_phi);
    printf("  acos(cos_phi) = %f\n", qlib::toDegree(acos(cos_phi)));
  }

  realnum_t lrkj = rkj.length();
  dPhidRi = rmj.scale( lrkj/(lrmj*lrmj));
  dPhidRl = rnk.scale(-lrkj/(lrnk*lrnk));
  
  realnum_t rijrkj = rij.dot(rkj);
  realnum_t rklrkj = rkl.dot(rkj);
  realnum_t lrkj2 = lrkj*lrkj;
  
  dPhidRj = dPhidRi.scale(rijrkj/lrkj2 - 1.0) - dPhidRl.scale(rklrkj/lrkj2);
  dPhidRk = dPhidRl.scale(rklrkj/lrkj2 - 1.0) - dPhidRi.scale(rijrkj/lrkj2);
  
  return phi;
}

#endif
