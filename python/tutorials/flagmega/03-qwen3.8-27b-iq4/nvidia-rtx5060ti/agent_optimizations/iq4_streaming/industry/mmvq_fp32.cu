// Adapted from llama.cpp 5ce06fc7 mmvq.cu/vecdotq.cuh. MIT: see LICENSE.
// Region baseline only; the production candidate remains Triton/TLE.
#include "vecdotq_fp32.cuh"
#include <cuda_runtime.h>

template<class Source>
static __device__ __forceinline__ float source_float(Source v) { return float(v); }
template<>
__device__ __forceinline__ float source_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<int D, int HALF>
static __device__ __forceinline__ float term(int packed, uint32_t h, uint32_t sl, const float *x) {
    const int s=threadIdx.x&7;
    const int t=HALF*128+s+D;
    const int word=16*(t/32)/4+(t%16)/4;
    const int aux_q4=__shfl_sync(0xffffffff,packed,word);
    return vec_dot_iq4_xs_fp32(aux_q4,h,sl,t);
}

template<int HALF>
static __device__ __forceinline__ float half_dot(int packed, uint32_t h, uint32_t sl, const float *x) {
    const int s=threadIdx.x&7;
    float v=__fmul_rn(term<8,HALF>(packed,h,sl,x),x[HALF*128+s+8]);
#define CONTRACT(D) v=__fmaf_rn(term<D,HALF>(packed,h,sl,x),x[HALF*128+s+D],v)
    CONTRACT(0); CONTRACT(16); CONTRACT(24); CONTRACT(32); CONTRACT(40);
#undef CONTRACT
#define ADD(D) v=__fadd_rn(__fmul_rn(term<D,HALF>(packed,h,sl,x),x[HALF*128+s+D]),v)
    ADD(48); ADD(56); ADD(64); ADD(72); ADD(80); ADD(88); ADD(96); ADD(104); ADD(112); ADD(120);
#undef ADD
    // Transport replaces upstream's split-K shared reduction to preserve
    // the elected virtual tree; every lane participates uniformly.
    float p0=__fadd_rn(__shfl_sync(0xffffffff,v,0),__shfl_sync(0xffffffff,v,1));
    float p1=__fadd_rn(__shfl_sync(0xffffffff,v,2),__shfl_sync(0xffffffff,v,3));
    float p2=__fadd_rn(__shfl_sync(0xffffffff,v,4),__shfl_sync(0xffffffff,v,5));
    float p3=__fadd_rn(__shfl_sync(0xffffffff,v,6),__shfl_sync(0xffffffff,v,7));
    return __fadd_rn(__fadd_rn(p0,p2),__fadd_rn(p1,p3));
}

static __device__ __forceinline__ float silu(float gate) {
    float e,recip;
    float neg=__fsub_rn(0.f,gate);
    float exponent=__fmul_rn(neg,__uint_as_float(0x3fb8aa3b));
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
    float denom=__fadd_rn(e,1.f);
    asm("div.full.f32 %0, %1, %2;" : "=f"(recip) : "f"(1.f), "f"(denom));
    return __fmul_rn(gate,recip);
}

// B1/no bias specialization of upstream mul_mat_vec_q. Retains its
// output-row grid, 2D warp block, full-K streaming and paired fusion arrays.
// The K-parallel lane map is replaced by row warps and virtual leaves.
template<class Source, bool has_fusion, int MODE>
__launch_bounds__(128,1)
static __global__ void mul_mat_vec_q(
        const void *vx_ptr, const Source *vy_ptr, const void *vgate,
        void *dst_ptr, void *second_ptr, const uint32_t ncols_x,
        const uint32_t stride_col_dst, const uint32_t stride_row_x) {
    const void * __restrict__ vx=vx_ptr;
    const Source * __restrict__ vy=vy_ptr;
    constexpr int ncols_dst=1,nwarps=4,rows_per_cuda_block=4,warp_size=32;
    const int tid=warp_size*threadIdx.y+threadIdx.x;
    const int row0=rows_per_cuda_block*blockIdx.x;
    const int blocks_per_row_x=ncols_x/256;
    const int row=row0+threadIdx.y;
    const bool active=uint32_t(row)<stride_col_dst;
    float tmp[ncols_dst]={0.f},tmp_gate[ncols_dst]={0.f};
    __shared__ float y[256];
    // Full-K remains serial at every output row. No DP4A or Q8 preparation.
    for(int kbx=0;kbx<blocks_per_row_x;++kbx) {
        y[tid]=source_float(vy[kbx*256+tid]);
        y[tid+128]=source_float(vy[kbx*256+tid+128]);
        const uint8_t *bq4=(const uint8_t *)vx+uint64_t(row)*stride_row_x+uint64_t(kbx)*136;
        int packed=active?get_int_b4(bq4+8,threadIdx.x):0;
        uint32_t header=(active&&threadIdx.x==0)?((const uint32_t*)bq4)[0]:0;
        uint32_t scales_l=(active&&threadIdx.x==0)?((const uint32_t*)bq4)[1]:0;
        header=__shfl_sync(0xffffffff,header,0);scales_l=__shfl_sync(0xffffffff,scales_l,0);
        int gp=0;uint32_t gh=0,gs=0;
        if constexpr(has_fusion) {
            const uint8_t *gb=(const uint8_t *)vgate+uint64_t(row)*stride_row_x+uint64_t(kbx)*136;
            gp=active?get_int_b4(gb+8,threadIdx.x):0;
            gh=(active&&threadIdx.x==0)?((const uint32_t*)gb)[0]:0;
            gs=(active&&threadIdx.x==0)?((const uint32_t*)gb)[1]:0;
            gh=__shfl_sync(0xffffffff,gh,0);gs=__shfl_sync(0xffffffff,gs,0);
        }
        __syncthreads();
        tmp[0]=__fadd_rn(tmp[0],half_dot<0>(packed,header,scales_l,y));
        tmp[0]=__fadd_rn(tmp[0],half_dot<1>(packed,header,scales_l,y));
        if constexpr(has_fusion) {
            tmp_gate[0]=__fadd_rn(tmp_gate[0],half_dot<0>(gp,gh,gs,y));
            tmp_gate[0]=__fadd_rn(tmp_gate[0],half_dot<1>(gp,gh,gs,y));
        }
        __syncthreads();
    }
    if(threadIdx.x==0 && active) {
        float result=tmp[0];
        if constexpr(has_fusion) result=__fmul_rn(silu(tmp[0]),tmp_gate[0]);
        if constexpr(MODE==0) ((float*)dst_ptr)[row]=result;
        if constexpr(MODE==1) ((__nv_bfloat16*)dst_ptr)[row]=__float2bfloat16_rn(result);
        if constexpr(MODE==2) {
            int address=(row/512)*256+row%256;
            __nv_bfloat16 *dst=(row%512)<256?(__nv_bfloat16*)dst_ptr:(__nv_bfloat16*)second_ptr;
            dst[address]=__float2bfloat16_rn(result);
        }
    }
}

extern "C" int iq4_industry_launch(const void *x,const void *w,const void *up,void *out,void *second,
        int n,int k,int source_f32,int mode,int glu,void *stream) {
    if(n<=0||k<=0||k%256||mode<0||mode>2||(mode==2&&n%512)) return -1;
    dim3 block(32,4,1),grid((n+3)/4,1,1);cudaStream_t s=(cudaStream_t)stream;
#define LAUNCH(S,G,M) mul_mat_vec_q<S,G,M><<<grid,block,0,s>>>(w,(const S*)x,up,out,second,k,n,(k/256)*136)
#define MODES(S) if(glu) { LAUNCH(S,true,1); } else if(mode==0) { LAUNCH(S,false,0); } else if(mode==1) { LAUNCH(S,false,1); } else { LAUNCH(S,false,2); }
    if(source_f32) { MODES(float); } else { MODES(__nv_bfloat16); }
#undef MODES
#undef LAUNCH
    return int(cudaGetLastError());
}
