#ifndef MOL_HPP_DEFINED
#define MOL_HPP_DEFINED

#include <qlib/LineStream.hpp>
#include <qlib/PrintStream.hpp>
#include <qlib/FileStream.hpp>
#include <qlib/Utils.hpp>

struct Bond
{
  int atom_i;
  int atom_j;
  float kf;
  float r0;
};

struct Angl
{
  int atom_i;
  int atom_j;
  int atom_k;
  float kf;
  float r0;
};

struct Dihe
{
  int atom_i;
  int atom_j;
  int atom_k;
  int atom_l;
  float kf;
  float r0;
  int npe;
};

struct Chir
{
  int atom_i;
  int atom_j;
  int atom_k;
  int atom_l;
  float kf;
  float r0;
  float fsgn;
};

struct PlanElem
{
  int iatom;
  float wgt;
};

struct Plan
{
  std::vector<PlanElem> atoms;
};

struct Rama
{
  int phi_ai;
  int phi_aj;
  int phi_ak;
  int phi_al;

  int psi_ai;
  int psi_aj;
  int psi_ak;
  int psi_al;
};

struct MolData
{
  int m_nAtoms;
  int m_nCrds;

  int m_nBonds;
  int m_nAngls;
  int m_nDihes;
  int m_nChirs;
  int m_nPlans;
  int m_nRamas;

  std::vector<float> m_crds;
  std::vector<float> m_mass;

  std::vector<Bond> m_bonds;
  std::vector<Angl> m_angls;
  std::vector<Dihe> m_dihes;
  std::vector<Chir> m_chirs;
  std::vector<Plan> m_plans;
  std::vector<Rama> m_ramas;

  std::vector<LString> m_pfx;

  void loadparm(const LString &fname)
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

  LString subStr2(int start, int end, const LString &str)
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

  void loadPDB(const LString &fname)
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

  void savePDB(const LString &fname)
  {
    qlib::FileOutStream fos;
    fos.open(fname);
    qlib::PrintStream ps(fos);

    int i;
    int natom = m_nAtoms;

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

  void addRand(float rng)
  {
    int i;
    const int natom = m_nAtoms;
    
    for (i=0; i<natom ;++i) {
      m_crds[i*3+0] += (drand48())*rng;
      m_crds[i*3+1] += (drand48())*rng;
      m_crds[i*3+2] += (drand48())*rng;
    }
  }

};





#endif
