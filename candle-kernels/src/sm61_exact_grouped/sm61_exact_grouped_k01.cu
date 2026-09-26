typedef unsigned long long flow_index_t;
#define N 1
#define CI 128
#define CO 128
#define GROUPS 4
#define CIG 32
#define COG 32
#define K 3
#define STRIDE 2
#define DILATION 1
#define PADDING 1
#define OUTPUT_PADDING 1
#define UNROLL_C 1
#define BLOCK_X 256
#define IN0 32
#define OUT0 64
#define TOTAL 8192

extern "C" __global__ __launch_bounds__(BLOCK_X) void flow_v0322_ct1d_s32_g4_u1_b256(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y){
  unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=TOTAL) return;
  unsigned ox=id%OUT0; unsigned oc=(id/OUT0)%CO; unsigned n=id/(OUT0*CO); unsigned group=oc/COG; unsigned ocg=oc%COG;
  float sum=0.0f;
  for(unsigned base=0;base<CIG;base+=UNROLL_C){
    #pragma unroll
    for(unsigned uu=0;uu<UNROLL_C;++uu){ unsigned icg=base+uu; if(icg<CIG){ unsigned ic=group*CIG+icg;
      #pragma unroll
      for(unsigned kk=0;kk<K;++kk){ int num=(int)ox+PADDING-(int)(kk*DILATION); if(num>=0 && num%STRIDE==0){ int ix=num/STRIDE; if(ix<IN0){
        flow_index_t xi=((flow_index_t)n*CI+ic)*IN0+(unsigned)ix; flow_index_t wi=((flow_index_t)ic*COG+ocg)*K+kk; sum=x[xi]*w[wi]+sum;
      }}}
    }}
  }
  y[id]=sum;
}
