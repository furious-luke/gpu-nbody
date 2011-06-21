#ifndef utils_hh
#define utils_hh

#include <assert.h>
#include <cuda.h>

#define CHKCERR(cec)				\
    assert(cec == cudaSuccess)

float randFloat( float low, float upp );

template<class T>
T min(T first, T second) {
    return (first < second) ? first : second;
}

#endif
