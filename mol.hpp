#ifndef MOL_HPP_DEFINED
#define MOL_HPP_DEFINED

#include <vector>

#include <qlib/LExceptions.hpp>
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

struct NonbElem
{
  int aj;
  float r0;
  float wgt;
};

struct Nonb
{
  std::vector<NonbElem> atoms;
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

  int m_nNonbPairMax;
  std::vector<Nonb> m_nonbs;

  std::vector<LString> m_pfx;

  void loadparm(const LString &fname);

  LString subStr2(int start, int end, const LString &str);

  void loadPDB(const LString &fname);

  void savePDB(const LString &fname);

#ifdef WIN32
  double drand48()
  {
    return ((double)(rand()) / RAND_MAX);
  }
#endif

  void addRand(float rng);

  void buildNonbData();

  bool isPolar(int ai) const;

  bool isBonded(int ai, int aj) const {
    if (ai>aj)
      std::swap(ai, aj);
    if (m_bondmap.find(std::pair<int,int>(ai, aj))==m_bondmap.end())
      return false;
    return true;
  }

  bool is1_4(int ai, int aj) const;

  std::set< std::pair<int,int> > m_bondmap;

  std::map<int, std::vector< std::pair<int, int> > > m_anglmap;

};





#endif
