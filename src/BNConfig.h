
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class BNConfig : public LayerConfig
{
public:
	size_t C = 96;
	size_t w = 4;
	size_t h = 4;
	size_t numBatches = 32;

	BNConfig(size_t _C, size_t _w, size_t _h, size_t _numBatches)
		:C(_C),
		w(_w),
		h(_h),
		  numBatches(_numBatches),
		  LayerConfig("BN"){};
};
