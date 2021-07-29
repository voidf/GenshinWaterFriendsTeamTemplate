#include "Headers.cpp"
#include "MathTheoryMisc.cpp"
#include "foreach.cpp"

/*
g 是mod(r*2^k+1)的原根
素数  r  k  g
3   1   1   2
5   1   2   2
17  1   4   3
97  3   5   5
193 3   6   5
257 1   8   3
7681    15  9   17
12289   3   12  11
40961   5   13  3
65537   1   16  3
786433  3   18  10
5767169 11  19  3
7340033 7   20  3
23068673    11  21  3
104857601   25  22  3
167772161   5   25  3
469762049   7   26  3
1004535809  479 21  3
2013265921  15  27  31
2281701377  17  27  3
3221225473  3   30  5
75161927681 35  31  3
77309411329 9   33  7
206158430209    3   36  22
2061584302081   15  37  7
2748779069441   5   39  3
6597069766657   3   41  5
39582418599937  9   42  5
79164837199873  9   43  5
263882790666241 15  44  7
1231453023109121    35  45  3
1337006139375617    19  46  3
3799912185593857    27  47  5
4222124650659841    15  48  19
7881299347898369    7   50  6
31525197391593473   7   52  3
180143985094819841  5   55  6
1945555039024054273 27  56  5
4179340454199820289 29  57  3
*/

/* 多项式 */
template <typename T>
struct Polynomial
{
    std::vector<T> cof; // 各项系数 coefficient 低次在前高次在后
    T mod = 998244353;  // 模数
    T G = 3;            // 原根
    T Gi = 332748118;   // 原根的逆元
    using pointval = std::pair<T, T>;
    std::vector<pointval> points; // x在前y在后
    Polynomial(T _mod) : mod(_mod) {}
    Polynomial() {}

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

    /* rev是蝴蝶操作数组，lim为填充到2的幂的值，mode为0正变换，1逆变换，逆变换后系数需要除以lim才是答案 */
    void NTT(std::vector<int> &rev, T lim, bool mode = 0)
    {
        for (auto i : range(lim))
            if (i < rev[i])
                swap(cof[i], cof[rev[i]]);
        for (T mid = 1; mid < lim; mid <<= 1)
        {
            T Wn = power(mode ? Gi : G, (mod - 1) / (mid << 1), mod);
            for (T j = 0; j < lim; j += mid << 1)
            {
                T w = 1;
                for (T k = 0; k < mid; ++k, w = (w * Wn) % mod)
                {
                    T x = cof[j + k], y = w * cof[j + k + mid] % mod;
                    cof[j + k] = madd(x, y); // 已经不得不用这个优化了
                    cof[j + k + mid] = msub(x, y);
                }
            }
        }
    }

    void _inv(T siz, Polynomial &B)
    {
        if (siz == 1)
        {
            B.cof[0] = inv(cof[0], mod);
            return;
        }
        _inv((siz + 1) >> 1, B);
        T lim = 1, limpow = 0;
        while (lim < (siz << 1))
            lim <<= 1, ++limpow;
        Polynomial C;
        C.cof.assign(cof.begin(), cof.begin() + siz);
        C.cof.resize(lim, 0);
        std::vector<int> rev(generateRev(lim, limpow));
        C.NTT(rev, lim, 0);
        B.NTT(rev, lim, 0);
        for (auto i : range(lim))
            B.cof[i] = msub(2LL , B.cof[i] * C.cof[i] % mod) * B.cof[i] % mod;
        B.NTT(rev, lim, 1);
        T iv = inv(lim, mod);
        for (auto i : range(siz))
            B.cof[i] = (B.cof[i] * iv) % mod;
        std::fill(B.cof.begin() + siz, B.cof.end(), 0);
    }

    Polynomial getinv()
    {
        T siz = cof.size();
        Polynomial B;
        T lim = 1, limpow = 0;
        while (lim < (siz << 1))
            lim <<= 1, ++limpow;
        B.cof.resize(lim, 0);
        cof.resize(lim, 0);
        B.cof.resize(siz);
        _inv(siz, B);
        return B;
    }

    Polynomial operator*(Polynomial &rhs)
    {
        return NTTMul(*this, rhs);
    }
    std::vector<int> generateRev(T lim, T limpow)
    {
        std::vector<int> rev(lim, 0);
        for (auto i : range(lim))
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (limpow - 1));
        return rev;
    }

    static inline void _mul(Polynomial &A, Polynomial &B, T lim, T limpow)
    {
        std::vector<int> rev(generateRev(lim, limpow));
        A.NTT(rev, lim, 0);
        B.NTT(rev, lim, 0);
        for (auto i : range(lim))
            A.cof[i] = (A.cof[i] * B.cof[i] % A.mod);
        A.NTT(rev, lim, 1);
    }
    /* 有点慢的NTT，换左值引用参数几乎没有优化 */
    static Polynomial NTTMul(Polynomial A, Polynomial B)
    {
        // assert(A.mod == B.mod);
        // Polynomial C(A.mod);
        T lim = 1;
        T limpow = 0;
        T retsiz = A.cof.size() + B.cof.size();
        while (lim <= retsiz)
            lim <<= 1, ++limpow;
        A.cof.resize(lim, 0);
        B.cof.resize(lim, 0);

        _mul(A, B, lim, limpow);

        A.cof.resize(retsiz - 1);
        T iv = inv(lim, A.mod);
        for (auto &i : A.cof)
            i = (i * iv) % A.mod;
        return A;
    }
};
