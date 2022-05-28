#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

#include <thread>
#include "Functionalities.h"

/************Debug****************/

// one party has the plain value and secret share to other
void debugPartySS(); // void funcPartySS
void debugPairRandom(); // void Precompute::getPairRand(Vec &a, size_t size)
void debugZeroRandom(); // void Precompute::getZeroShareRand(vector<T> &a, size_t size)
void debugReduction();
void debugPosWrap();
void debugWCExtension();
void debugBoolAnd();
void debugMixedShareGen();

void runTest(string str, string whichTest, string &network);

#endif