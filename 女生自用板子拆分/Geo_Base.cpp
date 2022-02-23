#ifndef Geo_Base_H
#define Geo_Base_H

#include "Headers.cpp"

// #include <cmath>
// #include <algorithm>
// #include <iostream>

namespace Geometry
{
    using fl = double;
    template <class T>
    T gcd(T a, T b) { return !b ? a : gcd(b, a % b); }

    constexpr const fl Infinity = INFINITY;
    const fl eps = 1e-8; // 精度参数

    const fl DEC = 1.0 / eps;

    int intereps(fl x)
    {
        if (x < -eps)
            return -1;
        else if (x > eps)
            return 1;
        return 0;
    }
    #define E		2.7182818284590452354
    #define LOG2E		1.4426950408889634074
    #define LOG10E	0.43429448190325182765
    #define LN2		0.69314718055994530942
    #define LN10		2.30258509299404568402
    #define PI		3.14159265358979323846
    #define PI_2		1.57079632679489661923
    #define PI_4		0.78539816339744830962
    // #define 1_PI		0.31830988618379067154
    // #define 2_PI		0.63661977236758134308
    // #define 2_SQRTPI	1.12837916709551257390
    #define SQRT2		1.41421356237309504880
    #define SQRT1_2	0.70710678118654752440

    bool round_compare(fl a, fl b) { return round(DEC * a) == round(DEC * b); }
    fl Round(fl a) { return round(DEC * a) / DEC; }

    /* 解一元二次方程，传出的x1为+delta，x2为-delta，如果无解返回两个nan */
    std::pair<fl, fl> solveQuadraticEquation(fl a, fl b, fl c)
    {
        fl delta = pow(b, 2) - 4 * a * c;
        if (delta < 0)
            return std::make_pair(nan(""), nan(""));
        else
        {
            delta = sqrt(delta);
            fl x1 = (-b + delta) / (2 * a);
            fl x2 = (-b - delta) / (2 * a);
            return std::make_pair(x1, x2);
        }
    }

    /* 
	求极大值，浮点型三分，实际上是假三分，接近二分复杂度
	思想：因为单峰，若极值在[ml, mr]左边，则必有f(ml)优于f(mr)，可以丢掉右端点
	若落在[ml, mr]内，随便丢一边都不会丢掉极值
	*/
	template <typename T>
	std::pair<fl, T> ternary_searchf(fl l, fl r, std::function<T(fl)> f, fl eps = 1e-6)
	{
		fl ee = eps / 3;
		while (l + eps < r)
		{
			fl mid = (l + r) / 2;
			fl ml = mid - ee;
			fl mr = mid + ee;
			if (f(ml) > f(mr)) // 改小于号变求极小值
				r = mr;
			else
				l = ml;
		}
		fl mid = (l + r) / 2;
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

#endif
