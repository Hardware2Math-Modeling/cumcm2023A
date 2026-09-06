// Optional native accelerator. The NumPy implementation is the readable oracle.
// No fast-math: preserve geometric boundary and energy-accounting semantics.
#include <cmath>
#include <algorithm>
#include <limits>

struct V { double x,y,z; };
inline V add(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V sub(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V mul(V a,double b){return {a.x*b,a.y*b,a.z*b};}
inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V load(const double* a,int i){return {a[3*i],a[3*i+1],a[3*i+2]};}
const double INF=std::numeric_limits<double>::infinity();

// First intersection with a CLOSED finite cylinder; kind 1=side, 2=cap.
// External receiver absorption accepts only kind 1, not a later side exit.
inline double cylinder(V p,V d,double tx,double ty,double radius,
                       double zlo,double zhi,int &kind){
    kind=0;
    if(radius<=0) return INF;
    double x=p.x-tx,y=p.y-ty,a=d.x*d.x+d.y*d.y;
    double best=INF;
    if(a>1e-20){
        double b=x*d.x+y*d.y,c=x*x+y*y-radius*radius;
        double disc=b*b-a*c;
        if(disc>=0){
            double root=std::sqrt(disc);
            for(int s=-1;s<=1;s+=2){
                double t=(-b+s*root)/a,z=p.z+t*d.z;
                if(t>1e-8 && z>=zlo-1e-9 && z<=zhi+1e-9 && t<best){best=t;kind=1;}
            }
        }
    }
    if(std::abs(d.z)>1e-15){
        for(int k=0;k<2;k++){
            double t=((k==0?zlo:zhi)-p.z)/d.z;
            double xx=x+t*d.x,yy=y+t*d.y;
            if(t>1e-8 && xx*xx+yy*yy<=radius*radius+1e-9 && t<best-1e-9){best=t;kind=2;}
        }
    }
    return best;
}

inline bool mirrors(V p,V d,int i,const long long* ptr,const long long* ids,
                    const double* centers,const double* normals,const double* us,
                    const double* vs,const double* widths,const double* heights){
    for(long long k=ptr[i];k<ptr[i+1];k++){
        int j=(int)ids[k];
        V n=load(normals,j);
        double denom=dot(d,n);
        if(std::abs(denom)<1e-13) continue;
        V delta=sub(p,load(centers,j));
        double t=-dot(delta,n)/denom;
        if(t<=1e-8) continue;
        V rel=add(delta,mul(d,t));
        if(std::abs(dot(rel,load(us,j)))<=widths[j]/2+1e-9 &&
           std::abs(dot(rel,load(vs,j)))<=heights[j]/2+1e-9) return true;
    }
    return false;
}

extern "C" void trace(int n,int samples,const double* centers,const double* widths,
    const double* heights,const double* normals,const double* us,const double* vs,
    const double* directions,const double* sample_uv,const double* gauss,
    const double* central_sun,const long long* sptr,const long long* sids,
    const long long* bptr,const long long* bids,double tx,double ty,double rz,
    double rr,double rh,double shaft,double slope,double* out){
    V sun=load(central_sun,0);
    double bottom=rz-rh/2,top=rz+rh/2;
    for(int i=0;i<n;i++){
        V c=load(centers,i),normal=load(normals,i),u=load(us,i),v=load(vs,i);
        double total=0,unshadowed=0,survived=0,accepted=0,isolated=0,blocked=0;
        for(int k=0;k<samples;k++){
            V s=load(directions,k);
            double weight=std::max(0.,dot(s,normal))/dot(s,sun);
            total+=weight;
            V p=add(c,add(mul(u,sample_uv[2*k]*widths[i]),mul(v,sample_uv[2*k+1]*heights[i])));
            V local=normal;
            if(slope>0){
                local=add(normal,add(mul(u,slope*gauss[2*k]),mul(v,slope*gauss[2*k+1])));
                local=mul(local,1/std::sqrt(dot(local,local)));
            }
            V r=sub(mul(local,2*dot(local,s)),s);
            int receive_kind,kind;
            double arrival=cylinder(p,r,tx,ty,rr,bottom,top,receive_kind);
            bool hit=(receive_kind==1);
            if(hit) isolated+=weight;
            // Upstream visibility from the mirror is +s, opposite photon travel.
            V incident=s;
            bool shadow=std::isfinite(cylinder(p,incident,tx,ty,shaft,0,bottom,kind));
            if(!shadow) shadow=std::isfinite(cylinder(p,incident,tx,ty,rr,bottom,top,kind));
            if(!shadow) shadow=mirrors(p,incident,i,sptr,sids,centers,normals,us,vs,widths,heights);
            if(shadow) continue;
            unshadowed+=weight;
            bool block=cylinder(p,r,tx,ty,shaft,0,bottom,kind)<arrival;
            if(!block) block=mirrors(p,r,i,bptr,bids,centers,normals,us,vs,widths,heights);
            if(block){blocked+=weight;continue;}
            survived+=weight;
            if(hit) accepted+=weight;
        }
        out[i*7]=total/samples;
        out[i*7+1]=total>0?survived/total:0;
        out[i*7+2]=survived>0?accepted/survived:0;
        out[i*7+3]=total>0?unshadowed/total:0;
        out[i*7+4]=total>0?blocked/total:0; // sequential blocking loss, no double counting
        out[i*7+5]=total>0?isolated/total:0;
        out[i*7+6]=accepted/samples;
    }
}
