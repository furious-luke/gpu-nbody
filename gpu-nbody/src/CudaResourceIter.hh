#ifndef CudaResourceIter_hh
#define CudaResourceIter_hh

#include "CudaDevice.hh"

namespace CUDA {

    class ResourceIter {

    public:

	ResourceIter();

	ResourceIter(const Device& device);

	~ResourceIter();

	void setDevice(const Device& device);

	void setBaseMemory(int baseGlMem);

	void setItemMetrics(int numItems, int shMemPerItem, int glMemPerItem, int threadsPerItem);

	void setSharedMemMetrics(int shMemPerThread, int shMemPerItem);

	void begin();

	bool done();

	void operator++();

	int getMaxItems() const;

	int getGridSize() const;

	int getBlockSize() const;

//  protected:

	void calcResources();
	int calcMaxItems();
	int calcMaxItemsBySharedMem();
	int calcMaxItemsByGlobalMem();
	int calcMaxItemsByThreads();
	int calcNumBlocks();
	int calcThreadsPerBlock();

	const Device* dev;
	int baseGlMem;
	int shMemPerItem;
	int glMemPerItem;
	int shMemPerThread;
	int glMemPerThread;
	int threadsPerItem;
	int shMemPerBlock;
	int glMemPerBlock;
	int nRemItems;
	int nItems;
	int nItemsPerIt;
	bool needRealloc;
	int maxItems;
	int nBlocks;
	int nThreads;

    };

}

#endif
