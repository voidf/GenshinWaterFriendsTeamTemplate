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