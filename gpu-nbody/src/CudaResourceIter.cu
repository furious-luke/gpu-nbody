#include <stdlib.h>
#include "CudaResourceIter.hh"
#include "utils.hh"

namespace CUDA {

    ResourceIter::ResourceIter() {
    }

    ResourceIter::ResourceIter(const Device& device)
	: dev((Device*)&device)
    {
    }

    ResourceIter::~ResourceIter() {
    }

    void ResourceIter::setDevice(const Device& device) {
	this->dev = &device;
    }

    void ResourceIter::setBaseMemory(int baseGlMem) {
	this->baseGlMem = baseGlMem;
    }

    void ResourceIter::setItemMetrics(int numItems, int shMemPerItem, int glMemPerItem, int threadsPerItem) {
	this->nItems = numItems;
	this->shMemPerItem = shMemPerItem;
	this->glMemPerItem = glMemPerItem;
	this->threadsPerItem = threadsPerItem;
    }

    void ResourceIter::setSharedMemMetrics(int shMemPerThread, int shMemPerItem) {
	this->shMemPerThread = shMemPerThread;
	this->shMemPerItem = shMemPerItem;
    }

    void ResourceIter::begin() {
	this->nRemItems = this->nItems;
	int oldMaxItems = this->maxItems;
	this->calcResources();
	if(this->maxItems != oldMaxItems)
	    this->needRealloc = true;
    }

    bool ResourceIter::done() {
	return !this->nRemItems || !this->maxItems;
    }

    void ResourceIter::operator++() {
	this->nRemItems -= this->maxItems;
	this->needRealloc = false;
	if(this->nRemItems < this->maxItems) {
	    int oldMaxItems = this->maxItems;
	    this->calcResources();
	    if(this->maxItems != oldMaxItems)
		this->needRealloc = true;
	}
    }

    int ResourceIter::getMaxItems() const {
	return this->maxItems;
    }

    int ResourceIter::getGridSize() const {
	return this->nBlocks;
    }

    int ResourceIter::getBlockSize() const {
	return this->nThreads;
    }

    void ResourceIter::calcResources() {
	assert(this->threadsPerItem >= 1);
	int numItemsPerBlockByThread = this->dev->getMaxThreadsPerBlock()/this->threadsPerItem;
	int numItemsPerBlockByShMem = this->shMemPerItem ? (this->dev->getSharedMemPerBlock()/this->shMemPerItem) : this->nRemItems;
	int numItemsPerBlock = min<int>(numItemsPerBlockByThread, numItemsPerBlockByShMem);
	int maxItemsByBlocks = numItemsPerBlock*this->dev->getNumBlocks();
	int maxItemsByGlMem = this->glMemPerItem ? (this->dev->getTotalGlobalMem()/this->glMemPerItem) : this->nRemItems;
	int maxItems = min<int>(min<int>(maxItemsByBlocks, maxItemsByGlMem), this->nRemItems);

	int numBlocks;
	if(numItemsPerBlock) {
	    numBlocks = maxItems/numItemsPerBlock;
	    if(!numBlocks) {
		numBlocks = 1;
		numItemsPerBlock = maxItems;
	    }
	}
	else
	    numBlocks = 0;

	int numThreadsPerBlock = numItemsPerBlock*this->threadsPerItem;

	this->maxItems = maxItems;
	this->nBlocks = numBlocks;
	this->nThreads = numThreadsPerBlock;
    }

}
