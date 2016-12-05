//
//
//


#ifndef RAMA_PLOT_DATA_HPP_INCLUDED
#define RAMA_PLOT_DATA_HPP_INCLUDED

#include <qlib/Vector2D.hpp>

class RamaPlotData
{
  typedef float value_t;
  typedef std::vector<value_t> data_t;

  int m_nWidth;
  // width == height

  data_t m_data;

public:

  RamaPlotData() : m_nWidth(0) {
  }

  void resize(int nwidth) {
    m_nWidth = nwidth;
    m_data.resize(m_nWidth * m_nWidth);
    fill(0.0);
  }

  void fill(value_t val) {
    const int nsize = m_data.size();
    for (int i=0; i<nsize; ++i)
      m_data[i] = val;
  }

  template <typename Type>
  void accum(const Type *psrc) {
    const int nsize = m_data.size();
    for (int i=0; i<nsize; ++i)
      m_data[i] += value_t(psrc[i]);
  }

  void normalize() {
    value_t sum = 0.0;
    const int nsize = m_data.size();
    for (int i=0; i<nsize; ++i)
      sum += m_data[i];
    const value_t scl = value_t(nsize) / (4.0*M_PI*M_PI*sum);
    for (int i=0; i<nsize; ++i)
      m_data[i] *= scl;
  }

  value_t getValue(int i, int j) const {
    return m_data[m_nWidth*i + j];
  }

  value_t getValuePeri(int i, int j) const {
    return getValue((i+m_nWidth)%m_nWidth,
		    (j+m_nWidth)%m_nWidth);
  }

  void setValue(int i, int j, value_t val) {
    m_data[m_nWidth*i + j] = val;
  }

  value_t getDensity(const qlib::Vector2D &pos) const;

  qlib::Vector2D getGrad(const qlib::Vector2D &pos) const;

  value_t getDensCubic(const qlib::Vector2D &pos) const;

  qlib::Vector2D getGradCubic(const qlib::Vector2D &pos) const;
  
  value_t getDensNrst(const qlib::Vector2D &pos) const;

  void setup();

  void dump(float delphi = 1.0) const;
};

#endif
