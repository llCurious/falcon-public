
#pragma once
#include "ReLUConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;


class ReLULayer : public Layer
{
private:
	ReLUConfig conf;
	ForwardVecorType activations;
	BackwardVectorType high_activations;
	BackwardVectorType deltas;
	RSSVectorSmallType reluPrime;

public:
	//Constructor and initializer
	ReLULayer(ReLUConfig* conf, int _layerNum);

	//Functions
	void printLayer() override;
	void forward(const ForwardVecorType& inputActivation) override;
	void computeDelta(BackwardVectorType& prevDelta) override;
	void updateEquations(const BackwardVectorType& prevActivations) override;

	// Mixed-precision funcs
	void weight_reduction() override;
	void activation_extension() override;
	void weight_extension() override;

	//Getters
	ForwardVecorType* getActivation() {return &activations;};
	BackwardVectorType* getHighActivation() {return &high_activations;};
	BackwardVectorType* getDelta() {return &deltas;};
};