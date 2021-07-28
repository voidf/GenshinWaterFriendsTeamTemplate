#include "Headers.cpp"
#include "MathTheoryMisc.cpp"
#include "foreach.cpp"

/* 多项式 */
template <typename T>
struct Polynomial
{
    std::vector<T> cof; // 各项系数 coefficient 低次在前高次在后
    T mod;
    using pointval = std::pair<T, T>;
    std::vector<pointval> points; // x在前y在后
    Polynomial(T _mod) : mod(_mod) {}

    /* n^2 拉格朗日插值 */
    void interpolation()
    {
        cof.assign(points.size(), 0);
        std::vector<T> num(cof.size() + 1, 0);
        std::vector<T> tmp(cof.size() + 1, 0);
        std::vector<T> invs(cof.size(), 0);
        num[0] = 1;
        for (int i = 1; i <= cof.size(); swap(num, tmp), ++i)
        {
            tmp[0] = 0;
            invs[i - 1] = inv(mod - points[i - 1].first, mod);
            for (auto j : range(1, i + 1))
                tmp[j] = num[j - 1];
            for (auto j : range(i + 1))
                modadd(tmp[j], num[j] * (mod - points[i - 1].first) % mod);
        }
        for (auto i : range(1, cof.size() + 1))
        {
            T den = 1, lst = 0;
            for (auto j : range(1, cof.size() + 1))
                if (i != j)
                    den = den * (points[i - 1].first - points[j - 1].first + mod) % mod;
            den = points[i - 1].second * inv(den) % mod;
            for (auto j : range(cof.size()))
            {
                tmp[j] = (num[j] - lst + mod) * invs[i - 1] % mod;
                modadd(cof[j], den * tmp[j] % mod), lst = tmp[j];
            }
        }
    }
    /* 计算多项式在x这点的值 */
    T eval(T x)
    {
        T ret = 0, px = 1;
        for (auto i : cof)
        {
            modadd(ret, i * px % mod);
            px = px * x % mod;
        }
        return ret;
    }
};
