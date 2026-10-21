// Standalone vector math, preserving the pinned CPU primitive operation order.
// SLEEF and MKL are independent libraries; no ATen/libtorch headers or calls.
#include <immintrin.h>
#include <sleef.h>
#include <mkl.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#if HDB2C13_LANES == 16
using Vec=__m512;
#define ENTRY(name) name ## _16
static Vec load(const float*x){return _mm512_loadu_ps(x);}
static void save(float*y,Vec x){_mm512_storeu_ps(y,x);}
static Vec fill(float x){return _mm512_set1_ps(x);}
static Vec add(Vec a,Vec b){return _mm512_add_ps(a,b);}
static Vec sub(Vec a,Vec b){return _mm512_sub_ps(a,b);}
static Vec mul(Vec a,Vec b){return _mm512_mul_ps(a,b);}
static Vec divv(Vec a,Vec b){return _mm512_div_ps(a,b);}
static Vec exponential(Vec x){return Sleef_expf16_u10avx512f(x);}
#else
using Vec=__m256;
#define ENTRY(name) name ## _8
static Vec load(const float*x){return _mm256_loadu_ps(x);}
static void save(float*y,Vec x){_mm256_storeu_ps(y,x);}
static Vec fill(float x){return _mm256_set1_ps(x);}
static Vec add(Vec a,Vec b){return _mm256_add_ps(a,b);}
static Vec sub(Vec a,Vec b){return _mm256_sub_ps(a,b);}
static Vec mul(Vec a,Vec b){return _mm256_mul_ps(a,b);}
static Vec divv(Vec a,Vec b){return _mm256_div_ps(a,b);}
static Vec exponential(Vec x){return Sleef_expf8_u10avx2(x);}
#endif
static constexpr int lanes=HDB2C13_LANES;
static Vec negative(Vec x){return sub(fill(0.0f),x);}

extern "C" int ENTRY(hdb2c13_unary)(int operation,int n,const float*x,float*y){
    if(n<=0 || !x || !y)return 1;
    constexpr MKL_INT64 mode=VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE;
    if(operation==0){vmsExp(n,x,y,mode);return 0;}
    if(operation==1){vmsSin(n,x,y,mode);return 0;}
    if(operation==2){vmsCos(n,x,y,mode);return 0;}
    if(operation!=3)return 2;
    int i=0;
    // Elementwise activation kernels vectorize pairs of vectors, then use
    // scalar libm for the remainder. Checkpoint FFN/channel widths are aligned.
    for(;i+2*lanes<=n;i+=2*lanes)for(int j=0;j<2;++j){
        Vec v=load(x+i+j*lanes);
        save(y+i+j*lanes,divv(v,add(fill(1.0f),exponential(negative(v)))));
    }
    for(;i<n;++i)y[i]=x[i]/(1.0f+std::exp(-x[i]));
    return 0;
}

extern "C" int ENTRY(hdb2c13_glu)(int batch,int channels,int time,const float*x,float*y){
    if(batch<=0 || channels<=0 || time<=0 || !x || !y)return 1;
    // Channel and time dimensions coalesce: each half is one contiguous
    // channels*time region. Scalar tails belong to that region, not each row.
    const int64_t n=int64_t(channels)*time;
    for(int b=0;b<batch;++b){
        const float *a=x+int64_t(b)*2*n;
        const float *gate=a+n;
        float *out=y+int64_t(b)*n;
        int64_t t=0;
        for(;t+2*lanes<=n;t+=2*lanes)for(int j=0;j<2;++j){
            Vec av=load(a+t+j*lanes),bv=load(gate+t+j*lanes);
            save(out+t+j*lanes,mul(av,divv(fill(1.0f),add(fill(1.0f),exponential(negative(bv))))));
        }
        for(;t<n;++t)out[t]=a[t]*(1.0f/(1.0f+std::exp(-gate[t])));
    }
    return 0;
}

static float reduce(const float*x,int n,bool maximum){
    if(n<lanes){
        float value=x[0];for(int i=1;i<n;++i)value=maximum?std::max(value,x[i]):value+x[i];
        return value;
    }
    float acc[lanes];std::memcpy(acc,x,sizeof(acc));int i=lanes;
    for(;i+lanes<=n;i+=lanes)for(int l=0;l<lanes;++l)
        acc[l]=maximum?std::max(acc[l],x[i+l]):acc[l]+x[i+l];
    for(int l=0;i+l<n;++l)acc[l]=maximum?std::max(acc[l],x[i+l]):acc[l]+x[i+l];
    // Same half/quarter/pair lane reduction order as the CPU vector kernel.
    for(int distance=lanes/2;distance;distance/=2){
        float next[lanes];for(int l=0;l<lanes;++l)
            next[l]=maximum?std::max(acc[l],acc[l^distance]):acc[l]+acc[l^distance];
        std::memcpy(acc,next,sizeof(acc));
    }
    return acc[0];
}
extern "C" int ENTRY(hdb2c13_softmax)(int rows,int n,const float*x,float*y){
    if(rows<=0 || n<=0 || !x || !y)return 1;
    for(int row=0;row<rows;++row){
        const float *in=x+int64_t(row)*n;float *out=y+int64_t(row)*n;
        const float max=reduce(in,n,true);int i=0;
        for(;i+lanes<=n;i+=lanes)save(out+i,exponential(sub(load(in+i),fill(max))));
        if(i<n){
            float tail[lanes]={};std::memcpy(tail,in+i,(n-i)*sizeof(float));
            save(tail,exponential(sub(load(tail),fill(max))));
            std::memcpy(out+i,tail,(n-i)*sizeof(float));
        }
        const float reciprocal=1.0f/reduce(out,n,false);
        for(int j=0;j<n;++j)out[j]=out[j]*reciprocal;
    }
    return 0;
}
