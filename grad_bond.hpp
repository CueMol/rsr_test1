#ifndef GRAD_BOND_HPP_INCLUDED
#define GRAD_BOND_HPP_INCLUDED

//#define DEBUG_PRINT 1

void gradBond(MolData *pMol, std::vector<float> &grad)
{
  int i;
  int nbond = pMol->m_nBonds;
  int ncrd = pMol->m_nAtoms * 3;

  //printf("target df nbond=%d, ncrd=%d\n", nbond, ncrd);
  float eBond = 0.0f;

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    float dx = pMol->m_crds[atom_i*3+0] - pMol->m_crds[atom_j*3+0];
    float dy = pMol->m_crds[atom_i*3+1] - pMol->m_crds[atom_j*3+1];
    float dz = pMol->m_crds[atom_i*3+2] - pMol->m_crds[atom_j*3+2];

    float sqlen = dx*dx + dy*dy + dz*dz;
    float len = sqrt(sqlen);

    float con = 2.0f * pMol->m_bonds[i].kf * (1.0f - pMol->m_bonds[i].r0/len);

    grad[atom_i*3+0] += con * dx;
    grad[atom_i*3+1] += con * dy;
    grad[atom_i*3+2] += con * dz;

    grad[atom_j*3+0] -= con * dx;
    grad[atom_j*3+1] -= con * dy;
    grad[atom_j*3+2] -= con * dz;

    float ss = len - pMol->m_bonds[i].r0;
    eBond += pMol->m_bonds[i].kf * ss * ss;
  }

#ifdef DEBUG_PRINT
  int natom = pMol->m_nAtoms;
  for (i=0; i<natom; ++i) {
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  printf("CPU Ebond = %f\n", eBond);
#endif
}

#include "cudacode.h"

CudaData *prepBondCuda(MolData *pMol)
{
  int i;
  int natom = pMol->m_nAtoms;
  int ncrd = natom * 3;

  CudaData *pRet = new CudaData();
  printf("prepCudaData nbond=%d, ncrd=%d\n", pMol->m_nBonds, ncrd);

  // setup exec layout
  if (natom<THR_PER_BLK) {
    pRet->nblk = 1;
    pRet->nthr = THR_PER_BLK;
  }
  else if (natom%THR_PER_BLK==0) {
    pRet->nblk = natom/THR_PER_BLK;
    pRet->nthr = THR_PER_BLK;
  }
  else {
    pRet->nblk = natom/THR_PER_BLK +1;
    pRet->nthr = THR_PER_BLK;
  }
  pRet->dev_natom = pRet->nblk * pRet->nthr;

  printf("nThr = %d (%d x %d)\n", pRet->dev_natom, pRet->nblk, pRet->nthr);

  ////////////////////////////////

  const int nbond = pMol->m_nBonds;
  std::vector<int> accum(natom);
  for (i=0; i<natom; ++i)
    accum[i] = 0;

  pRet->cubonds.resize(nbond);
  std::vector<CuBond> &cubonds = pRet->cubonds;

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    cubonds[i].ai = atom_i*3;
    cubonds[i].aj = atom_j*3;
    cubonds[i].kf = pMol->m_bonds[i].kf;
    cubonds[i].r0 = pMol->m_bonds[i].r0;

    //    cubonds[i].ai_ord = accum[atom_i];
    //    cubonds[i].aj_ord = accum[atom_j];

    accum[atom_i] ++;
    accum[atom_j] ++;
  }

  int nAccMax = 0;
  for (i=0; i<natom; ++i)
    nAccMax = qlib::max(nAccMax, accum[i]);

  printf("nAccMax = %d\n", nAccMax);
  pRet->nacc = nAccMax;

  for (i=0; i<natom; ++i)
    accum[i] = 0;

  //std::vector<int> bvec(natom*nAccMax);
  pRet->bvec.resize(pRet->dev_natom*nAccMax);
  std::vector<int> &bvec = pRet->bvec;

  for (i=0; i<bvec.size(); ++i)
    bvec[i] = 0;

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    int bind_i = accum[atom_i];
    accum[atom_i] ++;
    bvec[atom_i*nAccMax + bind_i] = (i+1);

    int bind_j = accum[atom_j];
    accum[atom_j] ++;
    bvec[atom_j*nAccMax + bind_j] = -(i+1);
    //bvec[atom_j*nAccMax + bind_j] = (i+1);
  }
  
#ifdef DEBUG_PRINT
  printf("bvec size=%d\n", bvec.size());
  for (i=0; i<natom; ++i) {
    printf("bvec %d: ", i);
    for (int j=0; j<nAccMax; ++j)
      printf("%d ", bvec[i*nAccMax + j]);
    printf("\n");
  }
#endif

  ////////////////////////////////
#if 0
  const int nangl = pMol->m_nAngls;
  for (i=0; i<natom; ++i)
    accum[i] = 0;
  pRet->cuangls.resize(nangl);
  std::vector<CuAngl> &cuangls = pRet->cuangls;

  for (i=0; i<nangl; i++) {
    int atom_i = pMol->m_angls[i].atom_i;
    int atom_j = pMol->m_angls[i].atom_j;
    int atom_k = pMol->m_angls[i].atom_k;

    cuangls[i].ai = atom_i*3;
    cuangls[i].aj = atom_j*3;
    cuangls[i].ak = atom_k*3;
    cuangls[i].kf = pMol->m_angls[i].kf;
    cuangls[i].r0 = pMol->m_angls[i].r0;

    accum[atom_i] ++;
    accum[atom_j] ++;

    accum[atom_k] ++;
    accum[atom_j] ++;
  }

  nAccMax = 0;
  for (i=0; i<natom; ++i)
    nAccMax = qlib::max(nAccMax, accum[i]);

  printf("Angl nAccMax = %d\n", nAccMax);
  pRet->nang_acc = nAccMax;

  for (i=0; i<natom; ++i)
    accum[i] = 0;

  pRet->ang_ivec.resize(pRet->dev_natom*nAccMax);
  std::vector<int> &avec = pRet->ang_ivec;
  
  for (i=0; i<avec.size(); ++i)
    avec[i] = 0;

  for (i=0; i<nangl; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    int bind_i = accum[atom_i];
    accum[atom_i] ++;
    bvec[atom_i*nAccMax + bind_i] = (i+1);

    int bind_j = accum[atom_j];
    accum[atom_j] ++;
    bvec[atom_j*nAccMax + bind_j] = -(i+1);
  }
#endif

  return pRet;
}


float gradBondCuda(MolData *pMol, CudaData *pDat, std::vector<float> &grad)
{
  
  float val;
  cudaBond_fdf(pMol->m_crds, pDat, &val, grad);

#ifdef DEBUG_PRINT
  int natom = pMol->m_nAtoms;
  for (int i=0; i<natom; ++i) {
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  printf("CUDA Ebond = %f\n", val);
#endif
  return val;
}

CudaData2 *prepBondCuda2(MolData *pMol)
{
  int i;
  int natom = pMol->m_nAtoms;
  int ncrd = natom * 3;

  CudaData2 *pRet = new CudaData2();

  printf("prepCudaData2 nbond=%d, natom=%d\n", pMol->m_nBonds, pMol->m_nAtoms);

  const int nbond = pMol->m_nBonds;
  std::vector<int> accum(natom);
  for (i=0; i<natom; ++i)
    accum[i] = 0;

  //std::vector<CuBond> cubonds(nbond);
  pRet->cubonds.resize(nbond);
  std::vector<CuBond> &cubonds = pRet->cubonds;

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    cubonds[i].ai = atom_i*3;
    cubonds[i].aj = atom_j*3;
    cubonds[i].kf = pMol->m_bonds[i].kf;
    cubonds[i].r0 = pMol->m_bonds[i].r0;

    //    cubonds[i].ai_ord = accum[atom_i];
    //    cubonds[i].aj_ord = accum[atom_j];

    accum[atom_i] ++;
    accum[atom_j] ++;
  }

  std::vector<int> idmap(natom);
  int nThr = 0;
  int iThr = 0;
  for (i=0; i<natom; ++i) {
    idmap[i] = nThr;
    nThr += accum[i];
    iThr += accum[i];
    if (i<natom-1) {
      if (iThr<THR_PER_BLK &&
	  iThr+accum[i+1]>THR_PER_BLK) {
	nThr += (THR_PER_BLK-iThr);
	iThr = 0;
      }
    }
    if (iThr==THR_PER_BLK)
      iThr = 0;
  }

  //for (i=0; i<natom; ++i) {
  //printf("Atom %d acc=%d--> %d\n", i, accum[i], idmap[i]);
  //}

  // setup exec layout
  if (nThr<THR_PER_BLK) {
    pRet->nblk = 1;
    pRet->nthr = THR_PER_BLK;
  }
  else if (nThr%THR_PER_BLK==0) {
    pRet->nblk = nThr/THR_PER_BLK;
    pRet->nthr = THR_PER_BLK;
  }
  else {
    pRet->nblk = nThr/THR_PER_BLK +1;
    pRet->nthr = THR_PER_BLK;
  }

  nThr = pRet->nblk * pRet->nthr;
  printf("nThr = %d (%d x %d)\n", nThr, pRet->nblk, pRet->nthr);

  pRet->bvec.resize(nThr * 2);
  std::vector<int> &bvec = pRet->bvec;

  for (i=0; i<natom; ++i)
    accum[i] = 0;
  for (i=0; i<bvec.size(); ++i)
    bvec[i] = 0;

  for (i=0; i<nbond; ++i) {
    int atom_i = pMol->m_bonds[i].atom_i;
    int atom_j = pMol->m_bonds[i].atom_j;

    int id_i = idmap[atom_i];
    int bind_i = accum[atom_i];
    accum[atom_i] ++;
    bvec[(id_i + bind_i)*2] = (i+1);

    int id_j = idmap[atom_j];
    int bind_j = accum[atom_j];
    accum[atom_j] ++;
    bvec[(id_j + bind_j)*2] = -(i+1);
  }
    
  for (i=0; i<natom; ++i) {
    int id = idmap[i];
    bvec[id*2+1] = i+1;
  }

  //  for (i=0; i<nThr; ++i) {
  //    printf("Thr %d: bond %d, atom %d\n", i, bvec[i*2], bvec[i*2+1]);
  //  }

  return pRet;
}


void gradBondCuda2(MolData *pMol, CudaData2 *pDat, std::vector<float> &grad)
{
  
  float val;
  cudaBond_fdf2(pMol->m_crds, pDat, &val, grad);

#ifdef DEBUG_PRINT
  int natom = pMol->m_nAtoms;
  for (int i=0; i<natom; ++i) {
    printf("%d (%f,%f,%f)\n", i, grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  }
  printf("CUDA Ebond = %f\n", val);
#endif

}


#endif
