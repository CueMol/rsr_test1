//
//
//

#include <common.h>
#include <qlib/LineStream.hpp>
#include <qlib/PrintStream.hpp>
#include <qlib/FileStream.hpp>
#include <qlib/Vector4D.hpp>

#include "mol.hpp"

using qlib::Vector4D;

void MolData::loadparm(const LString &fname)
{
  int i;
    
  qlib::FileInStream fis;
  fis.open(fname);
  qlib::LineStream ls(fis);
    
  LString line;
    
  line = ls.readLine().chomp();
  int natom;
  if (!line.toInt(&natom))
    throw qlib::RuntimeException();
    
  printf("NATOM=%d\n", natom);
  m_nAtoms = natom;
  int ncrd = natom*3;
  m_nCrds = ncrd;
  m_crds.resize(ncrd);
  m_mass.resize(ncrd);
  m_pfx.resize(natom);

  for (i=0; i<natom; ++i) {
    line = ls.readLine().chomp();
    // printf("line %d: %s\n", i, line.c_str());
    double val;
    if (!line.toDouble(&val))
      throw qlib::RuntimeException();
    m_mass[i] = float(val);
  }

  ///// BOND PARAMS /////

  line = ls.readLine().chomp();
  int nbond;
  if (!line.toInt(&nbond))
    throw qlib::RuntimeException();

  printf("NBOND=%d\n", nbond);
  m_bonds.resize(nbond);
  m_nBonds = nbond;

  std::list<LString> la;
  for (i=0; i<nbond; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    if (!iter->toInt(&m_bonds[i].atom_i))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_bonds[i].atom_j))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_bonds[i].kf))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_bonds[i].r0))
      throw qlib::RuntimeException();
    ++iter;
  }

  ///// ANGLE PARAMS /////

  line = ls.readLine().chomp();
  int nangl;
  if (!line.toInt(&nangl))
    throw qlib::RuntimeException();

  printf("NANGL=%d\n", nangl);
  m_angls.resize(nangl);
  m_nAngls = nangl;

  for (i=0; i<nangl; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    if (!iter->toInt(&m_angls[i].atom_i))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_angls[i].atom_j))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_angls[i].atom_k))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_angls[i].kf))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_angls[i].r0))
      throw qlib::RuntimeException();
    ++iter;

    //m_angls[i].r0 = qlib::toRadian(m_angls[i].r0);
  }

  ///// DIHEDRAL PARAMS /////

  line = ls.readLine().chomp();
  int ndihe;
  if (!line.toInt(&ndihe))
    throw qlib::RuntimeException();

  printf("NDIHE=%d\n", ndihe);
  m_dihes.resize(ndihe);
  m_nDihes = ndihe;

  for (i=0; i<ndihe; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    if (!iter->toInt(&m_dihes[i].atom_i))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_dihes[i].atom_j))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_dihes[i].atom_k))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_dihes[i].atom_l))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_dihes[i].kf))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_dihes[i].r0))
      throw qlib::RuntimeException();
    ++iter;
    // m_dihes[i].r0 = qlib::toRadian(m_dihes[i].r0);

    if (!iter->toInt(&m_dihes[i].npe))
      throw qlib::RuntimeException();
    ++iter;
  }

  ///// CHIRAL VOLUME PARAMS /////

  line = ls.readLine().chomp();
  int nchir;
  if (!line.toInt(&nchir))
    throw qlib::RuntimeException();

  printf("NCHIR=%d\n", nchir);
  m_chirs.resize(nchir);
  m_nChirs = nchir;

  for (i=0; i<nchir; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    if (!iter->toInt(&m_chirs[i].atom_i))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_chirs[i].atom_j))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_chirs[i].atom_k))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_chirs[i].atom_l))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_chirs[i].kf))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toRealNum<float>(&m_chirs[i].r0))
      throw qlib::RuntimeException();
    ++iter;

    int dum;
    if (!iter->toInt(&dum))
      throw qlib::RuntimeException();
    ++iter;
    m_chirs[i].fsgn = -1.0;
    if (dum) {
      // both sign
      if (m_chirs[i].r0<0)
	m_chirs[i].fsgn = 1.0;
    }

  }

  ///// PLANARITY PARAMS /////

  line = ls.readLine().chomp();
  int nplan;
  if (!line.toInt(&nplan))
    throw qlib::RuntimeException();

  printf("NPLAN=%d\n", nplan);
  m_plans.resize(nplan);
  m_nPlans = nplan;
    
  for (i=0; i<nplan; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    int natoms;
    if (!iter->toInt(&natoms))
      throw qlib::RuntimeException();
    ++iter;

    std::vector<PlanElem> &plan = m_plans[i].atoms;
    plan.resize(natoms);

    for (int j=0; j<natoms; ++j) {
      if (!iter->toInt(&plan[j].iatom))
	throw qlib::RuntimeException();
      ++iter;

      if (!iter->toRealNum<float>(&plan[j].wgt))
	throw qlib::RuntimeException();
      ++iter;
    }

  }

  ///// RAMAPLOT PARAMS /////

  line = ls.readLine().chomp();
  int nrama;
  if (!line.toInt(&nrama))
    throw qlib::RuntimeException();

  printf("NRAMA=%d\n", nrama);
  m_ramas.resize(nrama);
  m_nRamas = nrama;

  for (i=0; i<nrama; ++i) {
    line = ls.readLine().chomp();
    la.clear();
    line.split(' ', la);

    std::list<LString>::const_iterator iter = la.begin();

    if (!iter->toInt(&m_ramas[i].phi_ai))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].phi_aj))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].phi_ak))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].phi_al))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].psi_ai))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].psi_aj))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].psi_ak))
      throw qlib::RuntimeException();
    ++iter;

    if (!iter->toInt(&m_ramas[i].psi_al))
      throw qlib::RuntimeException();
    ++iter;
  }
 
}

LString MolData::subStr2(int start, int end, const LString &str)
{
  start --; end --;
  
  if (end >= str.length())
    end = str.length()-1;
  if (start<0)
    start = 0;
  if (start>end)
    start = end;
  
  return str.substr(start, end-start+1);
}

void MolData::loadPDB(const LString &fname)
{
  qlib::FileInStream fis;
  fis.open(fname);
  qlib::LineStream ls(fis);

  int i=0;
  int ncrds = m_nCrds;

  while (ls.ready() && i<ncrds) {
    LString line = ls.readLine().chomp();
    if (!line.startsWith("ATOM  "))
      continue;
    

    m_pfx[i/3] = subStr2(1, 30, line);

    if (!subStr2(31, 38, line).toRealNum<float>(&m_crds[i]))
      throw qlib::RuntimeException();
    ++i;

    if (!subStr2(39,46, line).toRealNum<float>(&m_crds[i]))
      throw qlib::RuntimeException();
    ++i;

    if (!subStr2(47,54, line).toRealNum<float>(&m_crds[i]))
      throw qlib::RuntimeException();
    ++i;

    //printf("read: %s\n", line.c_str());
  }
}

void MolData::savePDB(const LString &fname)
{
  qlib::FileOutStream fos;
  fos.open(fname);
  qlib::PrintStream ps(fos);

  int i;
  const int natom = m_nAtoms;

  for (i=0; i<natom ;++i) {
    ps.print(m_pfx[i]);
    ps.formatln(
		"%8.3f"
		"%8.3f"
		"%8.3f"
		"%6.2f"
		"%6.2f"
		"          "
		"    ",
		m_crds[i*3+0],
		m_crds[i*3+1],
		m_crds[i*3+2],
		0.0f,
		0.0f);
    //ps.println("");
  }
}

void MolData::addRand(float rng)
{
  int i;
  const int natom = m_nAtoms;
    
  for (i=0; i<natom ;++i) {
    m_crds[i*3+0] += (drand48())*rng;
    m_crds[i*3+1] += (drand48())*rng;
    m_crds[i*3+2] += (drand48())*rng;
  }
}

bool MolData::is1_4(int a1, int a4) const
{
  std::map<int, std::vector<std::pair<int,int> > >::const_iterator it_1, it_2;
  it_1 = m_anglmap.find(a1);
  if (it_1 == m_anglmap.end())
    return false;

  const std::vector<std::pair<int, int> > &v = it_1->second;
  for (int ii=0; ii<v.size(); ii++) {
    int idx_mid = v[ii].first;
    it_2 = m_anglmap.find(idx_mid);
    if (it_2 == m_anglmap.end())
      continue;

    const std::vector<std::pair<int, int> > &v_2 = it_2->second;
    for (int jj=0; jj<v_2.size(); jj++) { 
      if (v_2[jj].second == a4) {
	return true;
      }
    }

  }

  return false;
}

bool MolData::isPolar(int ai) const
{
  //???
  if (m_mass[ai]==12)
    return false;
  return true;
}

void MolData::buildNonbData()
{
  int i, j;
  const float cutoff = 10.0;
  const int natom = m_nAtoms;

  for (i=0; i<m_nBonds; ++i) {
    m_bondmap.insert(std::pair<int,int>(m_bonds[i].atom_i, m_bonds[i].atom_j));
  }

  for (i=0; i<m_nAngls; ++i) {
    int ai = m_angls[i].atom_i;
    int aj = m_angls[i].atom_j;
    int ak = m_angls[i].atom_k;

    m_bondmap.insert(std::pair<int,int>(ai, ak));

    m_anglmap[ai].push_back(std::pair<int,int>(aj, ak));
    m_anglmap[ak].push_back(std::pair<int,int>(aj, ai));
  }

  std::vector< std::deque<int> > szes(natom);

  for (i=0; i<natom ;++i) {
    Vector4D pos1(m_crds[i*3+0], m_crds[i*3+1], m_crds[i*3+2]);
    int isz = 0;
    for (j=i+1; j<natom ;++j) {

      Vector4D pos2(m_crds[j*3+0], m_crds[j*3+1], m_crds[j*3+2]);
      double dist = (pos1-pos2).length();
      if (dist>cutoff)
	continue;

      //avoid bonded pair
      if (isBonded(i,j))
	continue;

      //if (is1_4(i, j))
      //continue;

      //szes[i].push_back(j);
      //szes[j].push_back(i);
      //continue;

      //bool bcond = true;
      //bool bcond = drand48()<0.5;
      bool bcond = ((i+j)%2==0);

      if (bcond)
	szes[i].push_back(j);
      else
	szes[j].push_back(i);


      /*
      if (szes[i].size()<szes[j].size())
	szes[i].push_back(j);
      else if (szes[i].size()>szes[j].size())
	szes[j].push_back(i);
      else {
	bool bcond = ((i+j)%2==0);
	if (bcond)
	  szes[i].push_back(j);
	else
	  szes[j].push_back(i);
      }
      */

    }  
  }  


  int nsum = 0;
  int nmax = 0;
  for (i=0; i<natom ;++i) {
    int nsz = szes[i].size();
    // printf("Atom %d nonb size %d\n", i, nsz);
    nsum += nsz;
    nmax = qlib::max<int>(nmax, nsz);
  }
  printf("NONB nsum= %d\n", nsum);
  printf("NONB nmax= %d\n", nmax);
  printf("NONB nsum/natom= %d\n", nsum/natom);

  m_nNonbPairMax = nmax;

  m_nonbs.resize(natom);
  for (i=0; i<natom ;++i) {
    int nsz = szes[i].size();
    m_nonbs[i].atoms.resize(nsz);
    for (j=0; j<nsz; ++j) {
      m_nonbs[i].atoms[j].aj = szes[i][j];
      double r0 = 2.5;
      if (is1_4(i, j))
	r0 = 2.0;
      else if (isPolar(i) || isPolar(j)) {
	r0 = 2.3;
      }
      m_nonbs[i].atoms[j].r0 = 3.5; //r0;
      // ??? XXX
      //m_nonbs[i].atoms[j].wgt = 1.0/(0.02*0.02);
      m_nonbs[i].atoms[j].wgt = 1.0;
    }
  }

  m_bondmap.clear();
  m_anglmap.clear();
}


