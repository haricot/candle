#include <oneapi/dnnl/dnnl.hpp>
#include <omp.h>
#include <cstdio>
#include <unordered_map>

extern "C" int hdb2c11_conv2d(int batch,int ic,int ih,int iw,int oc,int kh,int kw,
    int stride,int pad,int dilation,int groups,int threads,
    const float *x,const float *w,const float *b,float *y,
    char *message,int message_size) {
    using namespace dnnl;
    try {
        omp_set_num_threads(threads);
        const int oh=(ih+2*pad-dilation*(kh-1)-1)/stride+1;
        const int ow=(iw+2*pad-dilation*(kw-1)-1)/stride+1;
        auto eng=engine(engine::kind::cpu,0);
        auto queue=stream(eng);
        memory::dims sd={batch,ic,ih,iw},dd={batch,oc,oh,ow};
        memory::dims wd=groups==1 ? memory::dims{oc,ic,kh,kw}
            : memory::dims{groups,oc/groups,ic/groups,kh,kw};
        auto user_s=memory::desc(sd,memory::data_type::f32,memory::format_tag::nchw);
        auto user_w=memory::desc(wd,memory::data_type::f32,
            groups==1?memory::format_tag::oihw:memory::format_tag::goihw);
        auto user_d=memory::desc(dd,memory::data_type::f32,memory::format_tag::nchw);
        auto bias=memory::desc({oc},memory::data_type::f32,memory::format_tag::a);
        primitive_attr attr;
        attr.set_fpmath_mode(fpmath_mode::strict);
        auto pd=convolution_forward::primitive_desc(eng,prop_kind::forward_training,
            algorithm::convolution_direct,
            memory::desc(sd,memory::data_type::f32,memory::format_tag::any),
            memory::desc(wd,memory::data_type::f32,memory::format_tag::any),
            bias,memory::desc(dd,memory::data_type::f32,memory::format_tag::any),
            {stride,stride},{dilation-1,dilation-1},{pad,pad},{pad,pad},attr);
        auto src_user=memory(user_s,eng,const_cast<float *>(x));
        auto weight_user=memory(user_w,eng,const_cast<float *>(w));
        auto dst_user=memory(user_d,eng,y);
        auto src=src_user;
        auto weight=weight_user;
        auto dst=dst_user;
        if (pd.src_desc()!=user_s) {
            src=memory(pd.src_desc(),eng);
            reorder(src_user,src).execute(queue,src_user,src);
        }
        if (pd.weights_desc()!=user_w) {
            weight=memory(pd.weights_desc(),eng);
            reorder(weight_user,weight).execute(queue,weight_user,weight);
        }
        if (pd.dst_desc()!=user_d) dst=memory(pd.dst_desc(),eng);
        auto bias_mem=memory(bias,eng,const_cast<float *>(b));
        convolution_forward(pd).execute(queue,{
            {DNNL_ARG_SRC,src},{DNNL_ARG_WEIGHTS,weight},
            {DNNL_ARG_BIAS,bias_mem},{DNNL_ARG_DST,dst}});
        if (pd.dst_desc()!=user_d) reorder(dst,dst_user).execute(queue,dst,dst_user);
        queue.wait();
        std::snprintf(message,message_size,"%s",pd.impl_info_str());
        return 0;
    } catch (const dnnl::error &e) {
        std::snprintf(message,message_size,"%s",e.what());return e.status;
    } catch (const std::exception &e) {
        std::snprintf(message,message_size,"%s",e.what());return -1;
    }
}
