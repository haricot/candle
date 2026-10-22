#include <mkl.h>
#include <string.h>

void hdb2c10_cpu_init(int threads, char *version, int length) {
    mkl_set_interface_layer(MKL_INTERFACE_LP64);
    mkl_set_threading_layer(MKL_THREADING_GNU);
    mkl_set_num_threads(threads);
    mkl_get_version_string(version,length);
}

long hdb2c10_cpu_rfft(int batch,int n,const float *input,float *output) {
    DFTI_DESCRIPTOR_HANDLE plan = 0;
    MKL_LONG status = DftiCreateDescriptor(&plan,DFTI_SINGLE,DFTI_REAL,1,(MKL_LONG)n);
    MKL_LONG strides[2] = {0,1};
    if (!status) status = DftiSetValue(plan,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    if (!status) status = DftiSetValue(plan,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)batch);
    if (!status) status = DftiSetValue(plan,DFTI_INPUT_DISTANCE,(MKL_LONG)n);
    if (!status) status = DftiSetValue(plan,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(n/2+1));
    if (!status) status = DftiSetValue(plan,DFTI_INPUT_STRIDES,strides);
    if (!status) status = DftiSetValue(plan,DFTI_OUTPUT_STRIDES,strides);
    if (!status) status = DftiSetValue(plan,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    if (!status) status = DftiCommitDescriptor(plan);
    if (!status) status = DftiComputeForward(plan,(void *)input,output);
    if (plan) DftiFreeDescriptor(&plan);
    return status;
}

void hdb2c10_cpu_addmm(int rows,int channels,int inner,const float *x,const float *w,
                      const float *bias,float *output) {
    for (int r=0;r<rows;r++) memcpy(output+(size_t)r*channels,bias,(size_t)channels*sizeof(float));
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,channels,inner,
                1.0f,x,inner,w,inner,1.0f,output,channels);
}

void hdb2c10_cpu_mel(int channels,int frames,int bins,const float *w,const float *x,float *out) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,channels,frames,bins,
                1.0f,w,bins,x,frames,0.0f,out,frames);
}

void hdb2c10_cpu_ln(int n,const float *x,float *out) { vsLn(n,x,out); }
void hdb2c10_cpu_sqrt(int n,const float *x,float *out) { vsSqrt(n,x,out); }
