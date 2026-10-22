// Independent scalar-lane implementation of the Welford/cascade FP32 order
// used in the pinned PyTorch CPU RowwiseMoments algorithm. No ATen dependency.
#include <cmath>
#include <algorithm>
#include <cstdint>

struct Moments {
    int64_t count=0;
    float mean[16]={};
    float variance[16]={};
};

static void combine(Moments &a,const Moments &b,int lanes) {
    const int64_t total=a.count+b.count;
    const float ratio=total ? float(b.count)/float(total) : 0.0f;
    for(int lane=0;lane<lanes;++lane){
        const float delta=b.mean[lane]-a.mean[lane];
        const float v=a.variance[lane]+b.variance[lane];
        const float scaled=ratio*delta;
        const float counted=delta*float(a.count);
        a.mean[lane]=a.mean[lane]+scaled;
        a.variance[lane]=std::fma(counted,scaled,v);
    }
    a.count=total;
}

static void row_moments(const float *x,int n,int lanes,float &mean,float &var){
    const int vectors=n/lanes;
    const int chunks=(vectors+15)/16;
    int depth=0;while((1<<depth)<chunks)++depth;
    Moments stack[32];
    for(int chunk=0;chunk<chunks;++chunk){
        Moments block;block.count=std::min(16,vectors-chunk*16);
        for(int j=0;j<block.count;++j){
            const float reciprocal=1.0f/float(j+1);
            for(int lane=0;lane<lanes;++lane){
                const float value=x[(chunk*16+j)*lanes+lane];
                const float delta=value-block.mean[lane];
                block.mean[lane]=std::fma(reciprocal,delta,block.mean[lane]);
                const float after=value-block.mean[lane];
                block.variance[lane]=std::fma(delta,after,block.variance[lane]);
            }
        }
        combine(stack[0],block,lanes);
        int mask=chunk+1;
        for(int level=1;level<depth && (mask&1)==0;++level){
            combine(stack[level],stack[level-1],lanes);
            stack[level-1]=Moments{};mask>>=1;
        }
    }
    for(int level=1;level<depth;++level)combine(stack[0],stack[level],lanes);
    int64_t count=0;mean=0.0f;var=0.0f;
    for(int i=vectors*lanes;i<n;++i){
        const float delta=x[i]-mean;
        ++count;mean+=delta/float(count);var+=delta*(x[i]-mean);
    }
    for(int lane=0;lane<lanes;++lane){
        const int64_t total=count+vectors;
        const float ratio=total ? float(vectors)/float(total) : 0.0f;
        const float delta=stack[0].mean[lane]-mean;
        mean+=ratio*delta;
        var+=stack[0].variance[lane]+delta*delta*ratio*float(count);
        count=total;
    }
    var/=float(n);
}

extern "C" int hdb2c12_layernorm(int rows,int n,int lanes,float epsilon,
    const float *x,const float *gamma,const float *beta,float *out){
    if(rows<=0 || n<=0 || lanes>16 || lanes<=0 || !x || !gamma || !beta || !out)return 1;
    for(int row=0;row<rows;++row){
        const float *input=x+(int64_t)row*n;float *output=out+(int64_t)row*n;
        float mean,var;row_moments(input,n,lanes,mean,var);
        const float rstd=1.0f/std::sqrt(var+epsilon);
        for(int col=0;col<n;++col)output[col]=(input[col]-mean)*rstd*gamma[col]+beta[col];
    }
    return 0;
}
