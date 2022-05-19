#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

/************Debug****************/

void debugPartySS(); // void funcShareSender(Vec &a, const vector<T> &data, const size_t size); void funcShareReceiver(Vec &a, const size_t size, const int shareParty)
void debugPairRandom(); // void Precompute::getPairRand(Vec &a, size_t size)
void debugReduction();


void runTest(string str, string whichTest, string &network);

#endif