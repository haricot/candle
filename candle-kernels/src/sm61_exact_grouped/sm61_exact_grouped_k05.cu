typedef unsigned long long flow_index_t;
#define N 1
#define CI 128
#define CO 128
#define GROUPS 32
#define CIG 4
#define COG 4
#define K 3
#define STRIDE 2
#define DILATION 1
#define PADDING 1
#define OUTPUT_PADDING 1
#define UNROLL_C 4
#define BLOCK_X 64
#define IN0 32
#define IN1 32
#define OUT0 64
#define OUT1 64
#define TOTAL 524288

extern "C" __global__ __launch_bounds__(BLOCK_X) void flow_v0322_ct2d_s32_g32_u4_b64(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y){
  unsigned id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=TOTAL) return;
  unsigned ox=id%OUT1; unsigned q=id/OUT1; unsigned oy=q%OUT0; q/=OUT0; unsigned oc=q%CO; unsigned n=q/CO; unsigned group=oc/COG; unsigned ocg=oc%COG;
  float sum=0.0f;
  for(unsigned base=0;base<CIG;base+=UNROLL_C){
    #pragma unroll
    for(unsigned uu=0;uu<UNROLL_C;++uu){ unsigned icg=base+uu; if(icg<CIG){ unsigned ic=group*CIG+icg;
      #pragma unroll
      for(unsigned ky=0;ky<K;++ky){ int ny=(int)oy+PADDING-(int)(ky*DILATION); if(ny<0 || ny%STRIDE!=0) continue; int iy=ny/STRIDE; if(iy>=IN0) continue;
        #pragma unroll
        for(unsigned kx=0;kx<K;++kx){ int nx=(int)ox+PADDING-(int)(kx*DILATION); if(nx>=0 && nx%STRIDE==0){ int ix=nx/STRIDE; if(ix<IN1){
          flow_index_t xi=(((flow_index_t)n*CI+ic)*IN0+(unsigned)iy)*IN1+(unsigned)ix; flow_index_t wi=(((flow_index_t)ic*COG+ocg)*K+ky)*K+kx; sum=x[xi]*w[wi]+sum;
        }}}
      }
    }}
  }
  y[id]=sum;
}
