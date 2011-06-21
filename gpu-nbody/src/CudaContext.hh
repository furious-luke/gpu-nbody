#ifndef CudaContext_hh
#define CudaContext_hh

namespace CUDA {

    class Context {

    public:

	Context();

	~Context();

	int getNumDevices() const;

//  protected:

    };

}

#endif
