
#ifndef MAP_HPP_INCLUDED
#define MAP_HPP_INCLUDED

#include <qlib/LineStream.hpp>
#include <qlib/PrintStream.hpp>
#include <qlib/FileStream.hpp>
#include <qlib/Matrix3D.hpp>

using qlib::Matrix3D;
using qlib::Vector4D;


typedef qlib::VectorND<3, int> Vector3I;

class DensityMap
{
public:
  /// cell dimensions
  double m_cella;
  double m_cellb;
  double m_cellc;
  double m_alpha;
  double m_beta;
  double m_gamma;
  
  double m_dScale;

  int m_stacol, m_starow, m_stasect;
  int m_endcol, m_endrow, m_endsect;
  int m_ncol, m_nrow, m_nsect;
  int m_na, m_nb, m_nc;

  // orth/frac matrix
  Matrix3D m_orthMat;
  Matrix3D m_fracMat;

  // density map
  std::vector<float> m_data;

  // loader tmp data
  int m_nFloatArrayCurPos;
  int m_nFloatArraySize;
  float m_floatArray[10];

  static LString readStrBlock(int bksz, int bkno, const LString instr) {
    return instr.substr(bkno*bksz, bksz);
  };

  void loadCNSMap(const LString &fname)
  {
    qlib::FileInStream fis;
    fis.open(fname);
    qlib::LineStream ins(fis);

    // skip the header remarks
    int ntit =-1;
    LString line;
    while (ins.ready()) {
      line = ins.readLine();

      LString stit = line.left(8).trim();
      if (stit.toInt(&ntit))
	break;
    }

    int i;
    for (i=0; i<ntit; i++) {
      line = ins.readLine();

      //if (startsWith(m_recbuf, " REMARKS")) {
      //LString msg = m_recbuf;
      //msg = msg.mid(9);
      //LOG_DPRINTLN("X-PLOR MapFile> %s",msg.c_str());
    }

    line = ins.readLine();
    readSectionInfo(line);

    line = ins.readLine();
    readCellInfo(line);

    line = ins.readLine();

    //
    //  allocate memory
    //
    int ntotal = m_ncol*m_nrow*m_nsect;
    m_data.resize(ntotal);
    printf("memory allocation %d elems\n", ntotal);
    
    m_nFloatArrayCurPos = 0;
    m_nFloatArraySize = 0;

    int ii=0;
    float rho;
    for (int isec=0; isec<m_nsect; isec++) {
      line = ins.readLine().trim();
      int ksec;
      
      // check section number
      if (!line.toInt(&ksec) || ksec!=isec) {
	throw qlib::FileFormatException();
      }
      
      // load one section
      for (int irow=0; irow<m_nrow; irow++) {
	for (int icol=0; icol<m_ncol; icol++) {
	  readDensity(ins, rho);
	  m_data[ii] = rho;
	  ii++;
	}
      }
    }

    printf("load Map (%d) OK\n", ii);
    setupMatrix();

    //m_dScale = 1.0;
    m_dScale = 60.0;
  }

  void readDensity(qlib::LineStream &ins, float &rho)
  {
    LString line;
    if (m_nFloatArrayCurPos>=m_nFloatArraySize) {
      line = ins.readLine();
      readFloatArray(line);
    }
    
    rho = m_floatArray[m_nFloatArrayCurPos];
    m_nFloatArrayCurPos ++;
  }

  void readFloatArray(const LString &line)
  {
    int nlen = line.length();
    m_nFloatArraySize = nlen/12;
    if (m_nFloatArraySize>6)
      throw qlib::FileFormatException();
    
    for (int i=0; i<m_nFloatArraySize; i++) {
      LString stmp = readStrBlock(12, i, line);
      float fdata;
      if (!stmp.toRealNum<float>(&fdata)) {
	throw qlib::FileFormatException();
      }
      m_floatArray[i] = fdata;
    }
    
    m_nFloatArrayCurPos = 0;
  }

  void readSectionInfo(const LString &line)
  {
    LString sNA, sAMIN, sAMAX, sNB, sBMIN, sBMAX, sNC, sCMIN, sCMAX;

    sNA = readStrBlock(8, 0, line);
    sAMIN = readStrBlock(8, 1, line);
    sAMAX = readStrBlock(8, 2, line);

    sNB = readStrBlock(8, 3, line);
    sBMIN = readStrBlock(8, 4, line);
    sBMAX = readStrBlock(8, 5, line);

    sNC = readStrBlock(8, 6, line);
    sCMIN = readStrBlock(8, 7, line);
    sCMAX = readStrBlock(8, 8, line);

    if (!sNA.toInt(&m_na) ||
	!sNB.toInt(&m_nb) ||
	!sNC.toInt(&m_nc) ||
	!sAMIN.toInt(&m_stacol) ||
	!sBMIN.toInt(&m_starow) ||
	!sCMIN.toInt(&m_stasect) ||
	!sAMAX.toInt(&m_endcol) ||
	!sBMAX.toInt(&m_endrow) ||
	!sCMAX.toInt(&m_endsect)) {
      throw qlib::FileFormatException();
    }

    m_ncol = m_endcol - m_stacol+1;
    m_nrow = m_endrow - m_starow+1;
    m_nsect = m_endsect - m_stasect+1;

    printf("na=%d\n", m_na);
  }

  void readCellInfo(const LString &line)
  {
    LString sa, sb, sc;
    LString salp, sbet, sgam;
    
    sa = readStrBlock(12, 0, line);
    sb = readStrBlock(12, 1, line);
    sc = readStrBlock(12, 2, line);
    
    salp = readStrBlock(12, 3, line);
    sbet = readStrBlock(12, 4, line);
    sgam = readStrBlock(12, 5, line);
    
    if (!sa.toDouble(&m_cella) ||
	!sb.toDouble(&m_cellb) ||
	!sc.toDouble(&m_cellc) ||
	!salp.toDouble(&m_alpha) ||
	!sbet.toDouble(&m_beta) ||
	!sgam.toDouble(&m_gamma)) {
      throw qlib::FileFormatException();
    }

    printf("cella=%f\n", m_cella);
    printf("cellb=%f\n", m_cellb);
    printf("cellc=%f\n", m_cellc);
    printf("alpha=%f\n", m_alpha);
    printf("beta=%f\n", m_beta);
    printf("gamma=%f\n", m_gamma);
  }

  void setupMatrix()
  {
    double alpha = m_alpha*M_PI/180;
    double beta = m_beta*M_PI/180;
    double gamma = m_gamma*M_PI/180;
    double coaster, siaster;

    coaster =
      (cos(beta)*cos(gamma) - cos(alpha))/
      (sin(beta)*sin(gamma));
    
    siaster = sqrt(1-coaster*coaster);
    
    m_orthMat.aij(1, 1) = m_cella;
    m_orthMat.aij(1, 2) = m_cellb * cos(gamma);
    m_orthMat.aij(1, 3) = m_cellc * cos(beta);
    
    m_orthMat.aij(2, 1) = 0.0;
    m_orthMat.aij(2, 2) = m_cellb * sin(gamma);
    m_orthMat.aij(2, 3) = -m_cellc * sin(beta)*coaster;
    
    m_orthMat.aij(3, 1) = 0.0;
    m_orthMat.aij(3, 2) = 0.0;
    m_orthMat.aij(3, 3) = m_cellc * sin(beta)*siaster;
    
    m_fracMat = m_orthMat.invert();
  }

  float getValue(const Vector3I &iv) const {
    return getValue(iv.ai(1), iv.ai(2), iv.ai(3));
  }

  float getValue(int i, int j, int k) const {
    if (0<=i && i<m_ncol &&
	0<=j && j<m_nrow &&
	0<=k && k<m_nsect)
      return m_data[i + (j + k*m_nrow)*m_ncol];
    else
      return 0.0f;
  }

  float getDensity(const Vector4D &pos) const
  {
    Vector4D frac = m_fracMat.mulvec(pos);
    float rho = 0.0;
    
    // conv to grid coordinates
    frac.x() *= double(m_na);
    frac.y() *= double(m_nb);
    frac.z() *= double(m_nc);

    // conv to map index
    frac.x() -= double(m_stacol);
    frac.y() -= double(m_starow);
    frac.z() -= double(m_stasect);

    Vector4D flpos(floor(frac.x()), floor(frac.y()), floor(frac.z()));
    
    Vector4D c1(frac.x()-flpos.x(), frac.y()-flpos.y(), frac.z()-flpos.z());
    Vector4D c0(1.0-c1.x(), 1.0-c1.y(), 1.0-c1.z());
    
    int ibase = int(flpos.x());
    int jbase = int(flpos.y());
    int kbase = int(flpos.z());
    
    float r00 = c0.z() * getValue(ibase, jbase, kbase);  // careful with evaluation order
    r00  += c1.z() * getValue(ibase, jbase,   kbase+1);// * map[ r.next_w() ];

    float r01 = c1.z() * getValue(ibase, jbase+1, kbase+1);// map[ r.next_v() ];
    r01  += c0.z() * getValue(ibase, jbase+1, kbase);//map[ r.prev_w() ];
    
    float r11 = c0.z() * getValue(ibase+1, jbase+1, kbase);//map[ r.next_u() ];
    r11  += c1.z() * getValue(ibase+1, jbase+1, kbase+1);//map[ r.next_w() ];

    float r10 = c1.z() * getValue(ibase+1, jbase, kbase+1);// map[ r.prev_v() ];
    r10  += c0.z() * getValue(ibase+1, jbase, kbase);// map[ r.prev_w() ];

    rho = ( c0.x()*( c0.y()*r00 + c1.y()*r01 ) + c1.x()*( c0.y()*r10 + c1.y()*r11 ) );
    
    return rho;
  }

  float getDensityCubic(const Vector4D &pos) const
  {
    Vector4D frac = m_fracMat.mulvec(pos);
    float rho = 0.0;
    
    // conv to grid coordinates
    frac.x() *= double(m_na);
    frac.y() *= double(m_nb);
    frac.z() *= double(m_nc);

    // conv to map index
    frac.x() -= double(m_stacol);
    frac.y() -= double(m_starow);
    frac.z() -= double(m_stasect);

    Vector4D flpos(floor(frac.x()), floor(frac.y()), floor(frac.z()));
    
    Vector4D c1(frac.x()-flpos.x(), frac.y()-flpos.y(), frac.z()-flpos.z());
    Vector4D c0(1.0-c1.x(), 1.0-c1.y(), 1.0-c1.z());
    
    // cubic spline coeffs
    /*
    float su, sv, sw,cu[4], cv[4], cw[4];
    cu[0] = -0.5*c1.x()*c0.x()*c0.x(); // cubic spline coeffs: u
    cu[1] = c0.x()*( -1.5*c1.x()*c1.x() + c1.x() + 1.0 );
    cu[2] = c1.x()*( -1.5*c0.x()*c0.x() + c0.x() + 1.0 );
    cu[3] = -0.5*c1.x()*c1.x()*c0.x();

    cv[0] = -0.5*c1.y()*c0.y()*c0.y(); // cubic spline coeffs: u
    cv[1] = c0.y()*( -1.5*c1.y()*c1.y() + c1.y() + 1.0 );
    cv[2] = c1.y()*( -1.5*c0.y()*c0.y() + c0.y() + 1.0 );
    cv[3] = -0.5*c1.y()*c1.y()*c0.y();

    cw[0] = -0.5*c1.z()*c0.z()*c0.z(); // cubic spline coeffs: u
    cw[1] = c0.z()*( -1.5*c1.z()*c1.z() + c1.z() + 1.0 );
    cw[2] = c1.z()*( -1.5*c0.z()*c0.z() + c0.z() + 1.0 );
    cw[3] = -0.5*c1.z()*c1.z()*c0.z();

    Vector3I iu;
    iu.ai(1) = int(flpos.x()) - 1;
    iu.ai(2) = int(flpos.y()) - 1;
    iu.ai(3) = int(flpos.z()) - 1;

    su = 0.0;
    int i, j;
    for ( j = 0; j < 4; j++ ) {
      Vector3I iv = iu;
      sv = 0.0;
      for ( i = 0; i < 4; i++ ) {
        Vector3I iw = iv;
	sw  = cw[0] * getValue(iw); //T( map[ iw ] );
	iw.ai(3) = iw.ai(3)+1;
	sw += cw[1] * getValue(iw); //T( map[ iw.next_w() ] );
	iw.ai(3) = iw.ai(3)+1;
	sw += cw[2] * getValue(iw); //T( map[ iw.next_w() ] );
	iw.ai(3) = iw.ai(3)+1;
	sw += cw[3] * getValue(iw); //T( map[ iw.next_w() ] );
        sv += cv[i] * sw;

        //iv.next_v();
	iv.ai(2) = iv.ai(2)+1;
      }
      su += cu[j] * sv;
      //iu.next_u();
      iu.ai(1) = iu.ai(1)+1;
    }
    */

    std::vector<Vector4D> co(4);

    //cu[0] = -0.5*cu1*cu0*cu0;
    co[0] = c1.scale(c0).scale(c0).scale(-0.5);

    //cu[1] = cu0*( -1.5*cu1*cu1 + cu1 + 1.0 );
    co[1] = c0.scale( c1.scale(c1).scale(-1.5).add(c1).add(1.0) );

    //cu[2] = cu1*( -1.5*cu0*cu0 + cu0 + 1.0 );
    co[2] = c1.scale( c0.scale(c0).scale(-1.5).add(c0).add(1.0) );

    //cu[3] = -0.5*cu1*cu1*cu0;
    co[3] = c1.scale(c1).scale(c0).scale(-0.5);

    double su = 0.0;
    int i, j, k;
    int ib = int(flpos.x()) - 1;
    int jb = int(flpos.y()) - 1;
    int kb = int(flpos.z()) - 1;

    for ( i = 0; i < 4; i++ ) {
      //iv = iu;
      double sv = 0.0;
      for ( j = 0; j < 4; j++ ) {
        //iw = iv;
	double sw = 0.0;
	for ( k = 0; k < 4; k++ ) {
	  sw += co[k].z() * getValue(ib+i, jb+j, kb+k);
	}
        sv += co[j].y() * sw;
        //iv.next_v();
      }
      su += co[i].x() * sv;
      //iu.next_u();
    }

    rho = float(su);
    return rho;
  }

  void getGrad(const Vector4D &pos, float &rval, Vector4D &grad) const
  {
    Vector4D frac = m_fracMat.mulvec(pos);
    
    // conv to grid coordinates
    frac.x() *= double(m_na);
    frac.y() *= double(m_nb);
    frac.z() *= double(m_nc);

    // conv to map index
    frac.x() -= double(m_stacol);
    frac.y() -= double(m_starow);
    frac.z() -= double(m_stasect);

    Vector4D flpos(floor(frac.x()), floor(frac.y()), floor(frac.z()));
    
    Vector4D c1(frac.x()-flpos.x(), frac.y()-flpos.y(), frac.z()-flpos.z());
    Vector4D c0(1.0-c1.x(), 1.0-c1.y(), 1.0-c1.z());
    
    // cubic spline coeffs
    std::vector<Vector4D> co(4);

    co[0] = c1.scale(c0).scale(c0).scale(-0.5);
    co[1] = c0.scale( c1.scale(c1).scale(-1.5).add(c1).add(1.0) );
    co[2] = c1.scale( c0.scale(c0).scale(-1.5).add(c0).add(1.0) );
    co[3] = c1.scale(c1).scale(c0).scale(-0.5);

    // cubic spline grad coeffs
    std::vector<Vector4D> go(4);
    //go[0] =  cu0*( 1.5*cu1 - 0.5 );
    go[0] = c0.scale( c1.scale(1.5).add(-0.5) );
    //go[1] =  cu1*( 4.5*cu1 - 5.0 );
    go[1] = c1.scale( c1.scale(4.5).add(-5.0) );
    //go[2] = -cu0*( 4.5*cu0 - 5.0 );
    go[2] = c0.scale( c0.scale(-4.5).add(5.0) );
    //go[3] = -cu1*( 1.5*cu0 - 0.5 );
    go[3] = c1.scale( c0.scale(-1.5).add(0.5) );

    double s1, s2, s3, du1, dv1, dv2, dw1, dw2, dw3;
    double rho;

    int i, j, k;
    int ib = int(flpos.x()) - 1;
    int jb = int(flpos.y()) - 1;
    int kb = int(flpos.z()) - 1;

    s1 = du1 = dv1 = dw1 = 0.0;
    for ( i = 0; i < 4; i++ ) {
      //iv = iu;
      s2 = dv2 = dw2 = 0.0;
      for ( j = 0; j < 4; j++ ) {
        //iw = iv;
	s3 = dw3 = 0.0;
	for ( k = 0; k < 4; k++ ) {
	  rho = getValue(ib+i, jb+j, kb+k);
	  s3 += co[k].z() * rho;
	  dw3 += go[k].z() * rho;
	}
        s2 += co[j].y() * s3;
	dv2 += go[j].y() * s3;
        dw2 += co[j].y() * dw3;
        //iv.next_v();
      }
      s1 += co[i].x() * s2;
      du1 += go[i].x() * s2;
      dv1 += co[i].x() * dv2;
      dw1 += co[i].x() * dw2;
      //iu.next_u();
    }
    rval = float(s1);
    grad = Vector4D(du1, dv1, dw1);

    grad.x() *= m_na;
    grad.y() *= m_nb;
    grad.z() *= m_nc;

    Matrix3D gradorth = m_fracMat.transpose();
    grad = gradorth.mulvec(grad);
    /*
    s1 = 0.0f;
    for ( i = 0; i < 4; i++ ) {
      for ( j = 0; j < 4; j++ ) {
	for ( k = 0; k < 4; k++ ) {
	  s1 += co[i].x() * co[j].y() * co[k].z() * getValue(ib+i, jb+j, kb+k);
	  printf("%d %d %d %e\n", i, j, k, co[i].x() * co[j].y() * co[k].z() * getValue(ib+i, jb+j, kb+k));
	}
      }
    }
    printf("sum=%f\n", s1);
    rval = float(s1);
    */
  }

  void getGradDescCubic(const Vector4D &pos, float &rval, Vector4D &grad) const
  {
    const double delta = 0.01;
    float ex0 = getDensityCubic(pos+Vector4D(-delta,0,0));
    float ex1 = getDensityCubic(pos+Vector4D(delta,0,0));
    float ey0 = getDensityCubic(pos+Vector4D(0,-delta,0));
    float ey1 = getDensityCubic(pos+Vector4D(0,delta,0));
    float ez0 = getDensityCubic(pos+Vector4D(0,0,-delta));
    float ez1 = getDensityCubic(pos+Vector4D(0,0,delta));
    
    grad = Vector4D((ex1-ex0)/(2*delta), (ey1-ey0)/(2*delta), (ez1-ez0)/(2*delta));
    rval = getDensityCubic(pos);
  }

  void getGradDesc(const Vector4D &pos, float &rval, Vector4D &grad) const
  {
    const double delta = 0.01;
    float ex0 = getDensity(pos+Vector4D(-delta,0,0));
    float ex1 = getDensity(pos+Vector4D(delta,0,0));
    float ey0 = getDensity(pos+Vector4D(0,-delta,0));
    float ey1 = getDensity(pos+Vector4D(0,delta,0));
    float ez0 = getDensity(pos+Vector4D(0,0,-delta));
    float ez1 = getDensity(pos+Vector4D(0,0,delta));
    
    grad = Vector4D((ex1-ex0)/(2*delta), (ey1-ey0)/(2*delta), (ez1-ez0)/(2*delta));
    rval = getDensity(pos);
  }
};

#endif
