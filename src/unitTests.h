#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

#include <thread>

/************Debug****************/

// one party has the plain value and secret share to other
void debugPartySS(); // void funcShareSender(Vec &a, const vector<T> &data, const size_t size); void funcShareReceiver(Vec &a, const size_t size, const int shareParty)
void debugPairRandom(); // void Precompute::getPairRand(Vec &a, size_t size)
void debugZeroRandom(); // void Precompute::getZeroShareRand(vector<T> &a, size_t size)
void debugReduction();
void debugPosWrap();
void debugWCExtension();

void runTest(string str, string whichTest, string &network);

#endif