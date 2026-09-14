typedef unsigned long long flow_index_t;
#define N 1
#define CI 64
#define CO 64
#define GROUPS 8
#define CIG 8
#define COG 8
#define K 3
#define STRIDE 1
#define DILATION 1
#define PADDING 1
#define OUTPUT_PADDING 0
#define UNROLL_C 1
#define BLOCK_X 256
#define IN0 128
#define OUT0 128
#define TOTAL 8192

extern "C" __global__ __launch_bounds__(BLOCK_X) void flow_v0322_gc1d_l128_g8_u1_b256(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y){
  unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=TOTAL) return;
  unsigned ox=id%OUT0; unsigned oc=(id/OUT0)%CO; unsigned n=id/(OUT0*CO); unsigned group=oc/COG;
  float sum=0.0f;
  for(unsigned base=0;base<CIG;base+=UNROLL_C){
    #pragma unroll
    for(unsigned uu=0;uu<UNROLL_C;++uu){ unsigned icg=base+uu; if(icg<CIG){ unsigned ic=group*CIG+icg;
      #pragma unroll
      for(unsigned kk=0;kk<K;++kk){ int ix=(int)(ox*STRIDE)+(int)(kk*DILATION)-PADDING; if(ix>=0 && ix<IN0){
        flow_index_t xi=((flow_index_t)n*CI+ic)*IN0+(unsigned)ix; flow_index_t wi=((flow_index_t)oc*CIG+icg)*K+kk; sum=x[xi]*w[wi]+sum;
      }}
    }}
  }
  y[id]=sum;
}
