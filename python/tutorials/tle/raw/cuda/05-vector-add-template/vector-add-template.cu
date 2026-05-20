
#define DEFINE_VECTOR_ADD(Type, FuncName)                              \
extern "C" __device__ void FuncName(                                   \
    __attribute__((address_space(1))) Type *C,                         \
    __attribute__((address_space(1))) const Type *A,                   \
    __attribute__((address_space(1))) const Type *B,                   \
    const int N) {                                                     \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;             \
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {           \
        C[i] = A[i] + B[i];                                            \
    }                                                                  \
}

#ifdef ENABLE_FLOAT
DEFINE_VECTOR_ADD(float,  vector_add_float)
#endif

#ifdef ENABLE_INT
DEFINE_VECTOR_ADD(int, vector_add_int)
#endif