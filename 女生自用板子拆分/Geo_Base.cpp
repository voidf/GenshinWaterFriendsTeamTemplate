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
    const double smooth_const2 = 0.479999989271164; // 二次项平滑系数
    const double smooth_const3 = 0.234999999403954; // 三次项平滑系数

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
}