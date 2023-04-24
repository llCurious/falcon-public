
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class BNRawConfig : public LayerConfig
{
public:
	size_t inputSize = 256;
	size_t numBatches = 32;

	BNRawConfig(size_t _D, size_t _numBatches)
		:inputSize(_D),
		  numBatches(_numBatches),
		  LayerConfig("BN"){};
};
