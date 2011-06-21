#include <stdlib.h>
#include <cuda.h>
#include "CudaContext.hh"
#include "CudaDevice.hh"
#include "utils.hh"

namespace CUDA {

    Context::Context() {
    }

    Context::~Context() {
    }

    int Context::getNumDevices() const {
	int numDevices;
	cudaError_t cec;
	cec = cudaGetDeviceCount( &numDevices ); CHKCERR( cec );
	return numDevices;
    }

}
