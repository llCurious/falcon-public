#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <tuple>
using namespace std;

#include "Precompute.h"
#include <thread>
#include "Functionalities.h"
#include <bitset>

extern Precompute *PrecomputeObject;

void mergeBoolVec(RSSVectorBoolType &result, const vector<bool> &a1, const vector<bool> &a2, size_t size);
void mergeRSSVectorBool(vector<bool> &result, RSSVectorBoolType &data, size_t size);
void funcBoolShareSender(RSSVectorBoolType &result, const vector<bool> &data, size_t size);
void funcBoolShareReceiver(RSSVectorBoolType &result, int shareParty, size_t size);

// void funcBoolShare(RSSVectorBoolType &result, const vector<bool> &a, size_t size);

void mergeBoolVec(RSSVectorBoolType &result, const vector<bool> &a1, const vector<bool> &a2, size_t size, string title, bool isprint);

void funcBoolAnd(RSSVectorBoolType &result, const RSSVectorBoolType &a1, const RSSVectorBoolType &a2, size_t size);

void funcBoolXor(RSSVectorBoolType &result, const RSSVectorBoolType &a1, const RSSVectorBoolType &a2, size_t size);

void funcBoolRev(vector<bool> &result, const RSSVectorBoolType &data, size_t size, string title, bool isprint);

template <typename Vec, typename T>
void funcB2AbyXOR(Vec &result, RSSVectorBoolType &data, size_t size);

template <typename Vec, typename T>
void funcB2A(Vec &result, const RSSVectorBoolType &data, size_t size);

// [b]^B = (b_0, b_1, b_2)
// P_A: (b_2, 0) (0, b_0)  (0, 0)
// P_B: (0, 0)   (b_0, 0)  (0, b_1)
// P_C: (0, b_2) (0, 0)    (b_1, 0)
template <typename Vec, typename T>
void funcB2AbyXOR(Vec &result, RSSVectorBoolType &data, size_t size)
{
    Vec aRss(size);
    Vec bRss(size);
    Vec amulb(size);
    if (partyNum == PARTY_A)
    {
        vector<T> a(size);
        for (size_t i = 0; i < size; i++)
        {
            a[i] = (data[i].first ^ data[i].second) ? 1 : 0;
        }

        funcShareSender<Vec, T>(aRss, a, size); // share a

        // TODO: precomute
        for (size_t i = 0; i < size; i++)
        {
            bRss[i] = make_pair(0, 0);
        }
    }
    else
    {
        funcShareReceiver<Vec, T>(aRss, size, PARTY_A); // get ss of a

        if (partyNum == PARTY_B)
        {
            for (size_t i = 0; i < size; i++)
            {
                bRss[i] = make_pair(0, data[i].second);
            }
        }
        else
        {
            for (size_t i = 0; i < size; i++)
            {
                bRss[i] = make_pair(data[i].first, 0);
            }
        }
    }

    // printRssVector<Vec>(aRss, "a_rss", size);
    // printRssVector<Vec>(bRss, "b_rss", size);

    // vector<T> a_plain(size);
    // vector<T> b_plain(size);

    // funcReconstruct<Vec, T>(aRss, a_plain, size, "a_plain", true);
    // funcReconstruct<Vec, T>(bRss, b_plain, size, "b_plain", true);

    // P_A: (a_2, a_0) (0,   0  )
    // P_B: (a_0, a_1) (0,   b_1)
    // P_C: (a_1, a_2) (b_1, 0  )

    // a ^ b1 = a + b1 - 2*a*b1
    funcDotProduct(aRss, bRss, amulb, size, false, FLOAT_PRECISION); // a * b1
    // printRssVector<Vec>(amulb, "a*b", size);

    // vector<T> amulb_plain(size);
    // funcReconstruct<Vec, T>(amulb, amulb_plain, size, "a*b_plain", true);

    for (size_t i = 0; i < size; i++)
    {
        result[i] = make_pair(aRss[i].first + bRss[i].first - 2 * amulb[i].first,
                              aRss[i].second + bRss[i].second - 2 * amulb[i].second);
    }
}

template <typename Vec, typename T>
void funcB2A(Vec &result, const RSSVectorBoolType &data, size_t size)
{
    RSSVectorBoolType randB(size);
    Vec randA(size);
    PrecomputeObject->getB2ARand(randB, size);
    funcB2AbyXOR<Vec, T>(ref(randA), ref(randB), size);
    // OFFLINE

    // vector<bool> rB(size);
    // funcBoolRev(rB, randB, size, "rand plain Bool", true);
    // vector<T> rA(size);
    // funcReconstruct<Vec, T>(randA, rA, size, "rand plain", true);
    // vector<bool> d(size);
    // funcBoolRev(d, data, size, "data plain Bool", true);

    RSSVectorBoolType cRss(size);
    funcBoolXor(cRss, data, randB, size); // mask data using rand
    vector<bool> c(size);
    funcBoolRev(c, cRss, size, "c plain", false);

    for (size_t i = 0; i < size; i++)
    {
        if (c[i])
        { // x1 = 1-c
            if (partyNum == PARTY_B)
            {
                result[i] = make_pair(-randA[i].first, 1 - randA[i].second);
            }
            else if (partyNum == PARTY_C)
            {
                result[i] = make_pair(1 - randA[i].first, -randA[i].second);
            }
            else
            {
                result[i] = make_pair(-randA[i].first, -randA[i].second);
            }
        }
        else
        {
            result[i] = make_pair(randA[i].first, randA[i].second);
        }
    }

    // vector<T> outputA(size);
    // funcReconstruct<Vec, T>(result, outputA, size, "output plain", true);

    // printRssVector(result, "b2a", size);
}