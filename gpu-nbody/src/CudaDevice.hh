#ifndef CudaDevice_hh
#define CudaDevice_hh

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "CudaContext.hh"

namespace CUDA {

    class Device {

    public:

	Device();

	Device(const Context& ctx);

	Device(const Context& ctx, int deviceIdx);

	~Device();

	void setContext(const Context& ctx);

	void setDeviceIndex(int deviceIdx);

	const cudaDeviceProp& getProperties() const;

	int getNumBlocks() const;

	int getTotalGlobalMem() const;

	int getSharedMemPerBlock() const;

	int getMaxThreadsPerBlock() const;

	void select() const;

	void print() const;

//  protected:

	Context* ctx;
	int deviceIdx;
	cudaDeviceProp prop;

    };

}

#endif
