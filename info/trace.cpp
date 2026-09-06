#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
constexpr double PI=3.14159265358979323846, INF=1e100;
struct V { double x,y,z; V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};} V operator-(V b)const{return{x-b.x,y-b.y,z-b.z};} V operator*(double a)const{return{x*a,y*a,z*a};} };
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm(V a){return sqrt(dot(a,a));} V unit(V a){return a*(1/norm(a));}
struct Mirror {V c,n,u,v,r; double w,h,rad,area,cosine,at;};
double radical(int n,int b){double x=0,f=1.0/b;while(n){x+=f*(n%b);n/=b;f/=b;}return x;}
uint64_t hash64(uint64_t z){z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
double rnd(uint64_t z){return (hash64(z)>>11)*0x1.0p-53;}
double frac(double a){return a-floor(a);}
bool plane_hit(V p,V d,const Mirror& m,double limit){double den=dot(d,m.n);if(abs(den)<1e-12)return false;double t=dot(m.c-p,m.n)/den;if(t<1e-7||t>limit)return false;V q=p+d*t-m.c;return abs(dot(q,m.u))<=m.w/2 && abs(dot(q,m.v))<=m.h/2;}
// First intersection with a closed opaque cylinder: 1=side, 2=cap, 0=none.
int cylinder(V p,V d,double tx,double ty,double radius,double low,double high,double& best){
 best=INF;if(radius<=0)return 0;double x=p.x-tx,y=p.y-ty;double a=d.x*d.x+d.y*d.y,b=2*(x*d.x+y*d.y),c=x*x+y*y-radius*radius;int kind=0;
 double disc=b*b-4*a*c;
 if(a>1e-14&&disc>=0){double q=sqrt(disc);double ts[2]={(-b-q)/(2*a),(-b+q)/(2*a)};for(double t:ts){double z=p.z+t*d.z;if(t>1e-7&&t<best&&z>=low&&z<=high){best=t;kind=1;}}}
 if(abs(d.z)>1e-14)for(double z:{low,high}){double t=(z-p.z)/d.z;double xx=x+t*d.x,yy=y+t*d.y;if(t>1e-7&&t<best&&xx*xx+yy*yy<=radius*radius){best=t;kind=2;}}
 return kind;
}
int main(int argc,char**argv){
 if(argc<5){cerr<<"trace design.txt detail.csv samples seed [solar_half_angle_rad=0.00465] [mast_radius=3.5] [threads=8]\n";return 1;}
 int ns=stoi(argv[3]);uint64_t seed=stoull(argv[4]);double alpha=argc>5?stod(argv[5]):.00465,mast=argc>6?stod(argv[6]):3.5;
 double probe_h=argc>8?stod(argv[8]):0,probe_z=argc>9?stod(argv[9]):0;
#ifdef _OPENMP
 omp_set_num_threads(argc>7?stoi(argv[7]):8);
#endif
 ifstream in(argv[1]);int N;double tx,ty;in>>N>>tx>>ty;if(!in||ns<1)return 2;vector<Mirror> m(N);double totalA=0;
 for(auto &e:m){in>>e.c.x>>e.c.y>>e.w>>e.h>>e.c.z;e.area=e.w*e.h;e.rad=.5*hypot(e.w,e.h);totalA+=e.area;double dist=norm(V{tx,ty,80}-e.c);e.r=unit(V{tx,ty,80}-e.c);e.at=.99321-.0001176*dist+1.97e-8*dist*dist;}if(!in)return 3;
 vector<array<double,4>> halton(ns);for(int k=0;k<ns;k++)halton[k]={radical(k+1,2),radical(k+1,3),radical(k+1,5),radical(k+1,7)};
 vector<double> annual(N,0),effannual(N,0);ofstream out(argv[2]);out<<setprecision(12)<<"month,time,dni,optical,cosine,shadow_block,intercept,atmospheric,power_MW,unit_kW_m2,optical_count,cosine_count,shadow_block_count,intercept_count\n";
 const int days[]={-59,-28,0,31,61,92,122,153,184,214,245,275};double lat=39.4*PI/180;double times[]={9,10.5,12,13.5,15};
 for(int month=0;month<12;month++)for(int ti=0;ti<5;ti++){
  double sd=sin(2*PI*days[month]/365)*sin(23.45*PI/180),cd=sqrt(1-sd*sd),omega=PI/12*(times[ti]-12);
  V s={-cd*sin(omega),cos(lat)*sd-sin(lat)*cd*cos(omega),sin(lat)*sd+cos(lat)*cd*cos(omega)};
  double dni=1.366*(.4237-.00821*9+(.5055+.00595*12.25)*exp(-(.2711+.01858*.25)/s.z));V su=unit(cross(V{0,0,1},s)),sv=cross(s,su);
  for(auto &e:m){e.n=unit(s+e.r);e.u=unit(cross(V{0,0,1},e.n));e.v=cross(e.n,e.u);e.cosine=dot(e.n,s);}
  vector<array<double,5>> result(N);
  #pragma omp parallel for schedule(dynamic,16)
  for(int i=0;i<N;i++){
   auto e=m[i];
   // Optional one-at-a-time geometry probes: all neighbours keep the input geometry.
   if(probe_h>0)e.h=min(e.w,probe_h);
   if(probe_z<0)e.c.z=max(2.,e.h/2+.05);else if(probe_z>0)e.c.z=max(probe_z,e.h/2+.05);
   if(probe_h>0||probe_z!=0){e.rad=.5*hypot(e.w,e.h);e.r=unit(V{tx,ty,80}-e.c);e.n=unit(s+e.r);e.u=unit(cross(V{0,0,1},e.n));e.v=cross(e.n,e.u);e.cosine=dot(e.n,s);double dist=norm(V{tx,ty,80}-e.c);e.at=.99321-.0001176*dist+1.97e-8*dist*dist;}
   vector<int> incoming,outgoing;
   for(int j=0;j<N;j++)if(i!=j){V d=m[j].c-e.c;double dd=dot(d,d),bound=e.rad+m[j].rad+alpha*sqrt(dd)*1.02;double a=dot(d,s),b=dot(d,e.r);if(a>-bound&&dd-a*a<bound*bound)incoming.push_back(j);if(b>-bound&&dd-b*b<bound*bound)outgoing.push_back(j);}
   double sh[4];for(int j=0;j<4;j++)sh[j]=rnd(seed+uint64_t(i+1)*0x9e3779b97f4a7c15ULL+uint64_t(month*5+ti+1)*0x632be59bd9b4e019ULL+j*0x94d049bb133111ebULL);
   int survive=0,hit=0;
   for(int k=0;k<ns;k++){
    V p=e.c+e.u*((frac(halton[k][0]+sh[0])-.5)*e.w)+e.v*((frac(halton[k][1]+sh[1])-.5)*e.h);
    double radius=tan(alpha)*sqrt(frac(halton[k][2]+sh[2])),az=2*PI*frac(halton[k][3]+sh[3]);V light=unit(s+su*(radius*cos(az))+sv*(radius*sin(az)));
    bool blocked=false;double tt;
    if(cylinder(p,light,tx,ty,3.5,76,84,tt))continue;
    if(mast>0&&cylinder(p,light,tx,ty,mast,0,76,tt))continue;
    for(int j:incoming)if(plane_hit(p,light,m[j],INF)){blocked=true;break;}if(blocked)continue;
    V reflected=e.n*(2*dot(e.n,light))-light;double rt;int kind=cylinder(p,reflected,tx,ty,3.5,76,84,rt);double limit=kind?rt:norm(V{tx,ty,80}-p)+10;
    for(int j:outgoing)if(plane_hit(p,reflected,m[j],limit)){blocked=true;break;}if(blocked)continue;
    if(mast>0&&cylinder(p,reflected,tx,ty,mast,0,76,tt)&&tt<limit)continue;
    survive++;if(kind==1)hit++;
   }
   double sb=double(survive)/ns,tr=survive?double(hit)/survive:0,opt=double(hit)/ns*e.cosine*e.at*.92;
   result[i]={opt,e.cosine,sb,tr,e.at};annual[i]+=dni*opt/60;effannual[i]+=opt/60;
  }
  array<double,5> sums{},counts{};double power=0;for(int i=0;i<N;i++){for(int k=0;k<5;k++){sums[k]+=result[i][k]*m[i].area;counts[k]+=result[i][k];}power+=dni*m[i].area*result[i][0];}
  out<<month+1<<','<<times[ti]<<','<<dni;for(double v:sums)out<<','<<v/totalA;out<<','<<power/1000<<','<<power/totalA;for(int k=0;k<4;k++)out<<','<<counts[k]/N;out<<'\n';
 }
 ofstream per(string(argv[2])+".mirrors.csv");per<<setprecision(12)<<"id,unit_kW_m2,optical\n";double p=0;for(int i=0;i<N;i++){per<<i+1<<','<<annual[i]<<','<<effannual[i]<<'\n';p+=annual[i]*m[i].area;}
 cout<<setprecision(10)<<"N="<<N<<" area="<<totalA<<" power_MW="<<p/1000<<" unit="<<p/totalA<<" samples="<<ns<<"\n";
}
