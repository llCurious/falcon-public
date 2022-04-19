
#ifndef PRECOMPUTE_H
#define PRECOMPUTE_H

#pragma once
#include "globals.h"


class Precompute
{
private:
	void initialize();

public:
	Precompute();
	~Precompute();

	template<typename Vec>
	void getDividedShares(Vec &r, Vec &rPrime, int d, size_t size);
	void getRandomBitShares(RSSVectorSmallType &a, size_t size);
	template<typename T>
	void getSelectorBitShares(RSSVectorSmallType &c, T &m_c, size_t size)
	{
		assert(c.size() == size && "size mismatch for getSelectorBitShares");
		assert(m_c.size() == size && "size mismatch for getSelectorBitShares");
		for(auto &it : c)
			it = std::make_pair(0,0);

		for(auto &it : m_c)
			it = std::make_pair(0,0);
	}

	template<typename T>
	void getShareConvertObjects(T &r, RSSVectorSmallType &shares_r, 
										RSSVectorSmallType &alpha, size_t size)
	{
		assert(shares_r.size() == size*BIT_SIZE && "getShareConvertObjects size mismatch");
		for(auto &it : r)
			it = std::make_pair(0,0);

		for(auto &it : shares_r)
			it = std::make_pair(0,0);

		for(auto &it : alpha)
			it = std::make_pair(0,0);
	}
	
	template<typename Vec>
	void getTriplets(Vec &a, Vec &b, Vec &c, 
					size_t rows, size_t common_dim, size_t columns);
	template<typename Vec>		
	void getTriplets(Vec &a, Vec &b, Vec &c, size_t size);
	void getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size);

};

template<typename Vec>
void Precompute::getDividedShares(Vec &r, Vec &rPrime, int d, size_t size) {
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

template<typename Vec>
void Precompute::getTriplets(Vec &a, Vec &b, Vec &c, 
				size_t rows, size_t common_dim, size_t columns) {
	assert(((a.size() == rows*common_dim) 
		and (b.size() == common_dim*columns) 
		and (c.size() == rows*columns)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}

template<typename Vec>		
void Precompute::getTriplets(Vec &a, Vec &b, Vec &c, size_t size) {
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}

// template void getDividedShares<RSSVectorHighType>(RSSVectorHighType &r, RSSVectorHighType &rPrime, int d, size_t size);
// template void getDividedShares<RSSVectorLowType>(RSSVectorLowType &r, RSSVectorLowType &rPrime, int d, size_t size);
// template void getTriplets<RSSVectorHighType>(RSSVectorHighType &a, RSSVectorHighType &b, RSSVectorHighType &c, size_t size);
// template void getTriplets<RSSVectorLowType>(RSSVectorLowType &a, RSSVectorLowType &b, RSSVectorLowType &c, size_t size);
#endif