#pragma once

#include "Headers.cpp"

// #include <cmath>
// #include <algorithm>
// #include <iostream>

namespace Geometry
{
    using FLOAT_ = double;
    template <class T>
    T gcd(T a, T b) { return !b ? a : gcd(b, a % b); }

    constexpr const FLOAT_ Infinity = INFINITY;
    const FLOAT_ decimal_round = 1e-8; // 精度参数

    const FLOAT_ DEC = 1.0 / decimal_round;

    int intereps(FLOAT_ x)
    {
        if (x < -decimal_round)
            return -1;
        else if (x > decimal_round)
            return 1;
        return 0;
    }

    const FLOAT_ PI = acos(-1);
    bool round_compare(FLOAT_ a, FLOAT_ b) { return round(DEC * a) == round(DEC * b); }
    FLOAT_ Round(FLOAT_ a) { return round(DEC * a) / DEC; }

    /* 解一元二次方程，传出的x1为+delta，x2为-delta，如果无解返回两个nan */
    std::pair<FLOAT_, FLOAT_> solveQuadraticEquation(FLOAT_ a, FLOAT_ b, FLOAT_ c)
    {
        FLOAT_ delta = pow(b, 2) - 4 * a * c;
        if (delta < 0)
            return std::make_pair(nan(""), nan(""));
        else
        {
            delta = sqrt(delta);
            FLOAT_ x1 = (-b + delta) / (2 * a);
            FLOAT_ x2 = (-b - delta) / (2 * a);
            return std::make_pair(x1, x2);
        }
    }

    /* 
	求极大值，浮点型三分，实际上是假三分，接近二分复杂度
	思想：因为单峰，若极值在[ml, mr]左边，则必有f(ml)优于f(mr)，可以丢掉右端点
	若落在[ml, mr]内，随便丢一边都不会丢掉极值
	*/
	template <typename T>
	std::pair<FLOAT_, T> ternary_searchf(FLOAT_ l, FLOAT_ r, std::function<T(FLOAT_)> f, FLOAT_ eps = 1e-6)
	{
		FLOAT_ ee = eps / 3;
		while (l + eps < r)
		{
			FLOAT_ mid = (l + r) / 2;
			FLOAT_ ml = mid - ee;
			FLOAT_ mr = mid + ee;
			if (f(ml) > f(mr)) // 改小于号变求极小值
				r = mr;
			else
				l = ml;
		}
		FLOAT_ mid = (l + r) / 2;
		return std::make_pair(mid, f(mid));
	}

    template <typename T>
	std::pair<LL, T> ternary_searchi(LL l, LL r, std::function<T(LL)> f)
	{
		while (l + 2 < r)
		{
			LL ml = l + r >> 1;
			LL mr = ml + 1;
			if (f(ml) < f(mr))
				r = mr;
			else
				l = ml;
		}
		std::pair<LL, T> ret = {l, f(l)};
		for (LL i = l + 1; i <= r; ++i)
		{
			T res = f(i);
			if (res < ret.second)
				ret = {i, res};
		}
		return ret;
	}

    /* GL用，如果输入合法可以逆时针序生成多边形一个三角剖分 */
    std::vector<unsigned int> generate_EBO(unsigned int polygonsize)
    {
        std::vector<unsigned int> V;
        V.reserve(3 * polygonsize);
        if (polygonsize)
        {
            unsigned int c = polygonsize;
            V.emplace_back(c);
            V.emplace_back(polygonsize - 1);
            V.emplace_back(0);

            for (unsigned i = 0; i < polygonsize - 1; ++i)
            {
                V.emplace_back(c);
                V.emplace_back(i);
                V.emplace_back(i + 1);
            }
        }
        return V;
    }
}