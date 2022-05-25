
#ifndef PRECOMPUTE_H
#define PRECOMPUTE_H

#pragma once
#include <string>
#include "globals.h"
#include "AESObject.h"
#include "connect.h"
#include "tools.h"
#include <thread>
using namespace std;

class Precompute
{
private:
	int partyNum;
	Precompute();
	void initialize();
	AESObject *aes_indep;
	AESObject *aes_next;
	AESObject *aes_prev;

public:
	Precompute(int partyNum, std::string basedir);
	~Precompute();

	template <typename Vec>
	void getDividedShares(Vec &r, Vec &rPrime, int d, size_t size);
	void getRandomBitShares(RSSVectorSmallType &a, size_t size);
	template <typename T>
	void getSelectorBitShares(RSSVectorSmallType &c, T &m_c, size_t size)
	{
		assert(c.size() == size && "size mismatch for getSelectorBitShares");
		assert(m_c.size() == size && "size mismatch for getSelectorBitShares");
		for (auto &it : c)
			it = std::make_pair(0, 0);

		for (auto &it : m_c)
			it = std::make_pair(0, 0);
	}

	template <typename T>
	void getShareConvertObjects(T &r, RSSVectorSmallType &shares_r,
								RSSVectorSmallType &alpha, size_t size)
	{
		size_t bit_size = BIT_SIZE;
		if (std::is_same<T, RSSVectorHighType>::value)
		{
			bit_size = BIT_SIZE_HIGH;
		}
		else if (std::is_same<T, RSSVectorLowType>::value)
		{
			bit_size = BIT_SIZE_LOW;
		}
		else
		{
			cout << "Not supported type" << typeid(r).name() << endl;
		}
		assert(shares_r.size() == size * bit_size && "getShareConvertObjects size mismatch");
		for (auto &it : r)
			it = std::make_pair(0, 0);

		for (auto &it : shares_r)
			it = std::make_pair(0, 0);

		for (auto &it : alpha)
			it = std::make_pair(0, 0);
	}

	template <typename Vec>
	void getTriplets(Vec &a, Vec &b, Vec &c,
					 size_t rows, size_t common_dim, size_t columns);
	template <typename Vec>
	void getTriplets(Vec &a, Vec &b, Vec &c, size_t size);
	void getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size);

	void getZeroBRand(vector<bool> &a, size_t size);
	void getZeroBShareSender(vector<RSSBoolType> &a, size_t size);
	void getZeroBSharePrev(vector<bool> &a, size_t size);
	void getZeroBShareReceiver(vector<RSSBoolType> &a, size_t size);

	template <typename T>
	void getNextRand(vector<T> &a, size_t size);

	template <typename T>
	void getPrevRand(vector<T> &a, size_t size);

	template <typename T>
	void getIndepRand(vector<T> &a, size_t size);

	template <typename Vec, typename T>
	void getPairRand(Vec &a, size_t size);

	template <typename Vec, typename T>
	void getZeroShareSender(Vec &a, size_t size);

	template <typename T>
	void getZeroSharePrev(vector<T> &a, size_t size);

	template <typename Vec, typename T>
	void getZeroShareReceiver(Vec &a, size_t size);

	// shareParty can know the all rand
	template <typename Vec, typename T>
	void getZeroShareRand(Vec &a, size_t size, int shareParty);

	template <typename Vec, typename T>
	void getZeroShareRand(vector<T> &a, size_t size);

	void getB2ARand(RSSVectorBoolType &dataB, size_t size);
};

template <typename T>
void Precompute::getNextRand(vector<T> &a, size_t size)
{
	assert(a.size() == size && "a.size is incorrect");

	for (size_t i = 0; i < size; i++)
	{
		a[i] = aes_next->getRand<T>();
	}
	// aes_next->getRand<T>(a);
}

template <typename T>
void Precompute::getPrevRand(vector<T> &a, size_t size)
{
	assert(a.size() == size && "a.size is incorrect");

	for (size_t i = 0; i < size; i++)
	{
		a[i] = aes_prev->getRand<T>();
	}
	// aes_prev->getRand<T>(a);
}

template <typename T>
void Precompute::getIndepRand(vector<T> &a, size_t size)
{
	assert(a.size() == size && "a.size is incorrect");

	aes_indep->getRand<T>(a);
}

template <typename Vec, typename T>
void Precompute::getPairRand(Vec &a, size_t size)
{
	assert(a.size() == size && "a.size is incorrect");

	for (size_t i = 0; i < size; i++)
	{
		a[i] = make_pair(aes_prev->getRand<T>(), aes_next->getRand<T>());
	}
}

// shareparty: a1, a2, (extra a3)
template <typename Vec, typename T>
void Precompute::getZeroShareSender(Vec &a, size_t size)
{
	vector<T> a3(size);
	thread receiver(receiveVector<T>, ref(a3), prevParty(partyNum), size); // receive a3

	vector<T> a2(size);
	getNextRand<T>(a2, size);

	receiver.join();

	for (size_t i = 0; i < size; i++)
	{
		T temp = 0 - a3[i] - a2[i];
		a[i] = make_pair(temp, a2[i]);
	}
}

// shareparty - 1 : a3
template <typename T>
void Precompute::getZeroSharePrev(vector<T> &a, size_t size)
{
	getPrevRand<T>(a, size);
	thread sender(sendVector<T>, ref(a), nextParty(partyNum), size);
	sender.join();
}

// shareparty + 1 : a2, a3
template <typename Vec, typename T>
void Precompute::getZeroShareReceiver(Vec &a, size_t size)
{
	getPairRand<Vec, T>(a, size);
}

/**
 * @brief work for three-party
 * shareparty     : a1, a2, (extra a3)
 * shareparty + 1 : a2, a3
 * shareparty - 1 : a3
 * a1 + a2 + a3 = 0
 *
 * @tparam T lowBit / HighBit
 * @tparam Vec randomness type
 * @param a save randomness
 * @param size
 * @param shareParty the party who has x
 */
template <typename Vec, typename T>
void Precompute::getZeroShareRand(Vec &a, size_t size, int shareParty)
{
	// assert(a.size() == size && "a.size is incorrect");

	// if (shareParty == partyNum)
	// {
	// 	vector<T> a3(size);
	// 	thread receiver(receiveVector<T>, ref(a3), nextParty(partyNum), size); // receive a3

	// 	vector<T> a2(size);
	// 	getNextRand<T>(a2, size);

	// 	receiver.join();

	// 	for (size_t i = 0; i < size; i++)
	// 	{
	// 		T temp = 0 - a3[i] - a2[i];
	// 		a[i] = make_pair(temp, a2[i]);
	// 	}
	// }
	// else if (nextParty(shareParty) == partyNum)
	// {
	// 	getPairRand<Vec, T>(a, size);
	// }
	// else
	// {
	// 	getPrevRand<T>(a, size);
	// 	thread sender(sendVector<T>, ref(a), prevParty(partyNum), size);
	// 	sender.join();
	// }
}

/**
 * @brief a0+a1+a2 = 0
 * every party only have one share of zero
 *
 * @tparam Vec
 * @tparam T
 * @param a
 * @param size
 */
template <typename Vec, typename T>
void Precompute::getZeroShareRand(vector<T> &a, size_t size)
{
	assert(a.size() == size && "a.size is incorrect");
	Vec paira(size);
	getPairRand<Vec, T>(paira, size);
	// printRssVector<Vec>(paira, "pair rand", size);

	for (size_t i = 0; i < size; i++)
	{
		a[i] = paira[i].first - paira[i].second;
	}
}

template <typename Vec>
void Precompute::getDividedShares(Vec &r, Vec &rPrime, int d, size_t size)
{
	assert(r.size() == size && "r.size is incorrect");
	assert(rPrime.size() == size && "rPrime.size is incorrect");

	for (int i = 0; i < size; ++i)
	{
		r[i].first = 0;
		r[i].second = 0;
		rPrime[i].first = 0;
		rPrime[i].second = 0;
		// r[i].first = 1;
		// r[i].second = 1;
		// rPrime[i].first = d;
		// rPrime[i].second = d;
	}
}

template <typename Vec>
void Precompute::getTriplets(Vec &a, Vec &b, Vec &c,
							 size_t rows, size_t common_dim, size_t columns)
{
	assert(((a.size() == rows * common_dim) and (b.size() == common_dim * columns) and (c.size() == rows * columns)) && "getTriplet size mismatch");

	for (auto &it : a)
		it = std::make_pair(0, 0);

	for (auto &it : b)
		it = std::make_pair(0, 0);

	for (auto &it : c)
		it = std::make_pair(0, 0);
}

template <typename Vec>
void Precompute::getTriplets(Vec &a, Vec &b, Vec &c, size_t size)
{
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");

	for (auto &it : a)
		it = std::make_pair(0, 0);

	for (auto &it : b)
		it = std::make_pair(0, 0);

	for (auto &it : c)
		it = std::make_pair(0, 0);
}

// template void getDividedShares<RSSVectorHighType>(RSSVectorHighType &r, RSSVectorHighType &rPrime, int d, size_t size);
// template void getDividedShares<RSSVectorLowType>(RSSVectorLowType &r, RSSVectorLowType &rPrime, int d, size_t size);
// template void getTriplets<RSSVectorHighType>(RSSVectorHighType &a, RSSVectorHighType &b, RSSVectorHighType &c, size_t size);
// template void getTriplets<RSSVectorLowType>(RSSVectorLowType &a, RSSVectorLowType &b, RSSVectorLowType &c, size_t size);
#endif