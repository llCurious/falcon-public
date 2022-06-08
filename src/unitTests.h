#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

#include <thread>
#include "Functionalities.h"

/************Debug****************/

// one party has the plain value and secret share to other
void debugPartySS();    // void funcPartySS
void debugPairRandom(); // void Precompute::getPairRand(Vec &a, size_t size)
void debugZeroRandom(); // void Precompute::getZeroShareRand(vector<T> &a, size_t size)
void debugReduction();
void debugPosWrap();
void debugWCExtension();
void debugBoolAnd();
void debugMixedShareGen();
void debugMSExtension(); // funcMSExtension(RSSVectorHighType &output, RSSVectorLowType &input, size_t size)
template <typename Vec, typename T>
void debugProbTruncation();
void debugTruncAndReduce();
template <typename Vec, typename T, typename RealVec>
void debugRandBit();
template <typename Vec, typename T>
void debugReciprocal();

void runTest(string str, string whichTest, string &network);

template <typename Vec, typename T, typename RealVec>
void debugRandBit()
{
    size_t size = 1000;

    RealVec b(size);
    vector<highBit> b_p(size);

    funcRandBit<Vec, T, RealVec>(b, size);

    // check
    funcReconstruct(b, b_p, size, "b plain", false);
    for (size_t i = 0; i < size; i++)
    {
        assert(b_p[i] == 0 || b_p[i] == 1);
    }
}

template <typename Vec, typename T>
void debugProbTruncation()
{
    int trunc_bits = 10;
    // high trunc
    size_t k = (sizeof(T) << 3) - 2;

    T temp = 1l << trunc_bits;

    size_t size = 1000;
    vector<T> data(size);
    size_t i = 0;

    data[i] = -(1l << (k));
    i++;
    for (; i < size / 4; i++)
    {
        data[i] = data[i - 1] + temp;
    }
    data[i] = 0;
    i++;
    for (; i < size / 2; i++)
    {
        data[i] = data[i - 1] - temp;
    }
    data[i] = 0;
    i++;
    for (; i < 3 * size / 4; i++)
    {
        data[i] = data[i - 1] + temp;
    }
    data[i] = (1l << k) - 1;
    i++;
    for (; i < size / 4; i++)
    {
        data[i] = data[i - 1] - temp;
    }

    int checkParty = PARTY_B;
    Vec datahigh(size);
    funcPartySS<Vec, T>(datahigh, data, size, checkParty);
    // above: test data

    Vec datatrunc(size);
    funcProbTruncation<Vec, T>(datatrunc, datahigh, trunc_bits, size);

    // check
#if (!LOG_DEBUG)
    vector<T> trunc_plain(size);
    funcReconstruct<Vec, T>(datatrunc, trunc_plain, size, "trunc plain", false);
    if (checkParty == partyNum)
    {
        if (k == 62)
        {
            for (size_t i = 0; i < size; i++)
            {
                // cout << (long)trunc_plain[i] << " " << (((long)data[i]) >> trunc_bits) << endl;
                long temp = ((long)data[i]) >> trunc_bits;
                assert(((long)trunc_plain[i] == temp) || ((long)trunc_plain[i] == temp + 1) || ((long)trunc_plain[i] == temp - 1));
                // cout << bitset<64>(trunc_plain[i]) << " " << bitset<64>((data[i] >> trunc_bits)) << endl;
            }
        }
        else
        {
            for (size_t i = 0; i < size; i++)
            {
                int temp = ((int)data[i]) >> trunc_bits;
                assert(((int)trunc_plain[i] == temp) || ((int)trunc_plain[i] == temp + 1) || ((int)trunc_plain[i] == temp - 1));
                // cout << (long)trunc_plain[i] << " " << (((long)data[i]) >> trunc_bits) << endl;
                // cout << bitset<64>(trunc_plain[i]) << " " << bitset<64>((data[i] >> trunc_bits)) << endl;
            }
        }
    }
#endif
}

template <typename Vec, typename T>
void debugReciprocal()
{
    size_t size = 5;
    vector<T> data(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (1 << (i + 14));
    }
    printVectorReal<T>(data, "input", size);

    Vec input(size);
    int checkParty = PARTY_B;
    funcPartySS(input, data, size, checkParty);

    Vec output(size);
    funcReciprocal(output, input, false, size);

    vector<T> result(size);
    funcReconstruct<Vec, T>(output, result, size, "out", true);
    printVectorReal<T>(result, "output", size);
}

#endif