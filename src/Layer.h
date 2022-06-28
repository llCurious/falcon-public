
#pragma once
#include "globals.h"

class Layer
{
public: 
	int layerNum = 0;
	Layer(int _layerNum): layerNum(_layerNum) {};

//Virtual functions	
	virtual void printLayer() {};
	virtual void forward(const ForwardVecorType& inputActivation) {};
	virtual void computeDelta(BackwardVectorType& prevDelta) {};
	virtual void updateEquations(const BackwardVectorType& prevActivations) {};

	// Mixed-Precision funcs
	virtual void weight_reduction() {};
	virtual void activation_extension() {};
	virtual void weight_extension() {};

//Getters
	virtual ForwardVecorType* getActivation() {};
	virtual BackwardVectorType* getHighActivation() {};
	virtual BackwardVectorType* getDelta() {};
};