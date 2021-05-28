namespace Geometry
{
    using FLOAT_ = double;

    constexpr const FLOAT_ Infinity = INFINITY;
    const FLOAT_ decimal_round = 1e-8; // 精度参数

    const FLOAT_ DEC = 1.0 / decimal_round;
    const double smooth_const2 = 0.479999989271164; // 二次项平滑系数
    const double smooth_const3 = 0.234999999403954; // 三次项平滑系数

    const FLOAT_ PI = acos(-1);
    bool round_compare(FLOAT_ a, FLOAT_ b) { return round(DEC * a) == round(DEC * b); }
    FLOAT_ Round(FLOAT_ a) { return round(DEC * a) / DEC; }
}