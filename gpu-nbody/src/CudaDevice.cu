#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "CudaDevice.hh"
#include "utils.hh"

namespace CUDA {

    Device::Device()
	: deviceIdx(-1)
    {
    }

    Device::Device(const Context& ctx)
	: ctx((Context*)&ctx),
	  deviceIdx(-1)
    {
    }

    Device::Device(const Context& ctx, int deviceIdx)
	: ctx((Context*)&ctx)
    {
	this->setDeviceIndex(deviceIdx);
    }

    Device::~Device() {
    }

    void Device::setContext(const Context& ctx) {
        this->ctx = (Context*)&ctx;
        this->deviceIdx = -1;
    }

    void Device::setDeviceIndex(int deviceIdx) {
        assert(this->ctx);
        assert(deviceIdx >= 0 && deviceIdx < this->ctx->getNumDevices());
        cudaError_t cec;
        this->deviceIdx = deviceIdx;
        cec = cudaGetDeviceProperties(&this->prop, deviceIdx); CHKCERR(cec);
    }

    const cudaDeviceProp& Device::getProperties() const {
        assert(this->deviceIdx >= 0);
        return this->prop;
    }

    int Device::getNumBlocks() const {
	assert(this->deviceIdx >= 0);
	return this->prop.maxGridSize[0]*this->prop.maxGridSize[1]*this->prop.maxGridSize[2];
    }

    int Device::getTotalGlobalMem() const {
	assert(this->deviceIdx >= 0);
	return this->prop.totalGlobalMem;
    }

    int Device::getSharedMemPerBlock() const {
        assert(this->deviceIdx >= 0);
        return this->prop.sharedMemPerBlock;
    }

    int Device::getMaxThreadsPerBlock() const {
	assert(this->deviceIdx >= 0);
	return this->prop.maxThreadsPerBlock;
    }

    void Device::select() const {
        assert(this->deviceIdx >= 0);
        cudaError_t cec;
        cec = cudaSetDevice(this->deviceIdx); CHKCERR(cec);
    }

    void Device::print() const {
        assert(this->deviceIdx >= 0);
        printf("Device name = %s\n", this->prop.name);
        printf("Number of blocks = %d\n", this->prop.multiProcessorCount);
        printf("Threads per block = %d\n", this->prop.maxThreadsPerBlock);
        printf("Thread dim size = (%d, %d, %d)\n",
               this->prop.maxThreadsDim[0], this->prop.maxThreadsDim[1], this->prop.maxThreadsDim[2]);
        printf("Registers per block = %d\n", this->prop.regsPerBlock);
        printf("Shared memory per block = %d\n", this->prop.sharedMemPerBlock);
        printf("Total constant memory = %ld\n", this->prop.totalConstMem);
        printf("Total global memory = %ld\n", this->prop.totalGlobalMem);
        printf("Warp size = %d\n", this->prop.warpSize);
    }

}
