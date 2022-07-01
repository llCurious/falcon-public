#include "globals.h"
#if (USE_GPU)
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>


#include <stdio.h> 
#include <iostream>
#include "Config.h"
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>
#include <time.h>
#include "secCompMultiParty.h"
#include "main_gf_funcs.h"
#include <string>
#include <openssl/sha.h>
#include <math.h>
#include <sstream>
#include "AESObject.h"
#include "connect.h"

// template<typename Vec, typename T>
void matrixMultRSS_Cuda(const RSSVectorHighType &a, const RSSVectorHighType &b, vector<highBit> &temp3, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);
void matrixMultRSS_Cuda(const RSSVectorLowType &a, const RSSVectorLowType &b, vector<lowBit> &temp3, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);
#endif