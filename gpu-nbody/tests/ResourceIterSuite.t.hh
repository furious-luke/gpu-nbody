#include <limits.h>
#include <cxxtest/TestSuite.h>
#include "gpu-nbody/CudaResourceIter.hh"

using namespace CUDA;

class ResourceIterSuite : public CxxTest::TestSuite {

public:

    void setUp() {
	this->dev.setContext(this->ctx);
	this->dev.setDeviceIndex(0);
	this->ri.setDevice(this->dev);
	this->numBlocks = this->dev.getNumBlocks();
	this->glMem = this->dev.getTotalGlobalMem();
	this->shMemPerBlock = this->dev.getSharedMemPerBlock();
	this->numThreadsPerBlock = this->dev.getMaxThreadsPerBlock();
    }

    void tearDown() {
    }

    void testNoItems() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(0, 1, 1, 1);
	for(ri.begin(); !ri.done(); ++ri) {
	    TS_FAIL("Should not iterate.");
	}
    }

    void testInsufShMem() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(1, LONG_MAX, 1, 1);
	for(ri.begin(); !ri.done(); ++ri) {
	    TS_FAIL("Should not iterate.");
	}
    }

    void testInsufGlMem() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(1, 1, LONG_MAX, 1);
	for(ri.begin(); !ri.done(); ++ri) {
	    TS_FAIL("Should not iterate.");
	}
    }

    void testInsufThreads() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(1, 1, 1, LONG_MAX);
	for(ri.begin(); !ri.done(); ++ri) {
	    TS_FAIL("Should not iterate.");
	}
    }

    void testTrivial() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(1, 0, 0, 1);
	int numIts = 0;
	for(ri.begin(); !ri.done(); ++ri, ++numIts) {
	    TS_ASSERT_EQUALS(ri.getMaxItems(), 1);
	    TS_ASSERT_EQUALS(ri.getGridSize(), 1);
	    TS_ASSERT_EQUALS(ri.getBlockSize(), 1);
	}
	TS_ASSERT_EQUALS(numIts, 1);
    }

    void testOneBlock() {
	ResourceIter& ri = this->ri;
	ri.setItemMetrics(this->numThreadsPerBlock, 0, 0, 1);
	int numIts = 0;
	for(ri.begin(); !ri.done(); ++ri, ++numIts) {
	    TS_ASSERT_EQUALS(ri.getMaxItems(), this->numThreadsPerBlock);
	    TS_ASSERT_EQUALS(ri.getGridSize(), 1);
	    TS_ASSERT_EQUALS(ri.getBlockSize(), this->numThreadsPerBlock);
	}
	TS_ASSERT_EQUALS(numIts, 1);
    }

    void testAllBlocks() {
	ResourceIter& ri = this->ri;
	int numAllThreads = this->numThreadsPerBlock*this->numBlocks;
	ri.setItemMetrics(numAllThreads, 0, 0, 1);
	int numIts = 0;
	for(ri.begin(); !ri.done(); ++ri, ++numIts) {
	    TS_ASSERT_EQUALS(ri.getMaxItems(), numAllThreads);
	    TS_ASSERT_EQUALS(ri.getGridSize(), this->numBlocks);
	    TS_ASSERT_EQUALS(ri.getBlockSize(), this->numThreadsPerBlock);
	}
	TS_ASSERT_EQUALS(numIts, 1);
    }

    void testTwiceAllBlocks() {
	ResourceIter& ri = this->ri;
	int numAllThreads = this->numThreadsPerBlock*this->numBlocks;
	ri.setItemMetrics(2*numAllThreads, 0, 0, 1);
	int numIts = 0;
	for(ri.begin(); !ri.done(); ++ri, ++numIts) {
	    TS_ASSERT_EQUALS(ri.getMaxItems(), numAllThreads);
	    TS_ASSERT_EQUALS(ri.getGridSize(), this->numBlocks);
	    TS_ASSERT_EQUALS(ri.getBlockSize(), this->numThreadsPerBlock);
	}
	TS_ASSERT_EQUALS(numIts, 2);
    }

    void testDoubleThreads() {
	ResourceIter& ri = this->ri;
	int numAllThreads = this->numThreadsPerBlock*this->numBlocks;
	ri.setItemMetrics(2*numAllThreads, 0, 0, 2);
	int numIts = 0;
	for(ri.begin(); !ri.done(); ++ri, ++numIts) {
	    TS_ASSERT_EQUALS(ri.getMaxItems(), numAllThreads/2);
	    TS_ASSERT_EQUALS(ri.getGridSize(), this->numBlocks);
	    TS_ASSERT_EQUALS(ri.getBlockSize(), this->numThreadsPerBlock);
	}
	TS_ASSERT_EQUALS(numIts, 4);
    }

//protected:

    Context ctx;
    Device dev;
    ResourceIter ri;
    int numBlocks;
    int glMem;
    int shMemPerBlock;
    int numThreadsPerBlock;

};
