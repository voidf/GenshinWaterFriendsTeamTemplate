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
    LL mod = 998244353; // 模数
    LL G = 3;           // 原根
    LL Gi = 332748118;  // 原根的逆元
    using pointval = std::pair<T, T>;
    std::vector<pointval> points; // x在前y在后

    inline LL modadd(LL &x, LL y) { return (x += y) >= mod ? x -= mod : x; }
    inline LL madd(LL x, LL y) { return (x += y) >= mod ? x - mod : x; }
    inline LL modsub(LL &x, LL y) { return (x -= y) < 0 ? x += mod : x; }
    inline LL msub(LL x, LL y) { return (x -= y) < 0 ? x + mod : x; }

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
    void NTT(std::vector<int> &rev, LL lim, bool mode = 0)
    {
        int l;
        for (int i = 0; i < lim; ++i)
            if (i < rev[i])
                swap(cof[i], cof[rev[i]]);
        for (int mid = 1; mid < lim; mid = l)
        {
            l = mid << 1;
            T Wn = power(mode ? Gi : G, (mod - 1) / (mid << 1), mod);
            for (int j = 0; j < lim; j += l)
            {
                T w = 1;
                for (int k = 0; k < mid; ++k, w = ((LL)w * Wn) % mod)
                {
                    T x = cof[j | k], y = (LL)w * cof[j | k | mid] % mod;
                    cof[j | k] = madd(x, y); // 已经不得不用这个优化了
                    cof[j | k | mid] = msub(x, y);
                }
            }
        }
        if (mode)
        {
            T iv = inv(lim, mod);
            for (auto &i : cof)
                i = ((LL)i * iv) % mod;
        }
    }

    /* 精度更高的写法 */
    void FFT(std::vector<int> &rev, LL lim, bool mode, std::vector<T> &Wn)
    {
        LL &n = lim;
        if (mode)
            for (int i = 1; i < n; i++)
                if (i < (n - i))
                    std::swap(cof[i], cof[n - i]);
        for (int i = 0; i < n; i++)
            if (i < rev[i])
                std::swap(cof[i], cof[rev[i]]);

        for (int m = 1, l = 0; m < n; m <<= 1, l++)
        {
            for (int i = 0; i < n; i += m << 1)
            {
                for (int k = i; k < i + m; k++)
                {
                    T W = Wn[1ll * (k - i) * n / m];
                    T a0 = cof[k], a1 = cof[k + m] * W;
                    cof[k] = a0 + a1;
                    cof[k + m] = a0 - a1;
                }
            }
        }
        if (mode)
            for (auto &i : cof)
                i /= lim;
    }

    /* 建议模数满足原根时使用，1e5 O2 331ms，无O2 612ms, 写成循环只优化了空间 */
    void N_inv(int siz, Polynomial &B)
    {
        B.cof.emplace_back(inv(cof[0], mod));
        LL bas = 2, lim = 4, limpow = 2;
        Polynomial A;
        while (bas < (siz << 1))
        {
            B.cof.resize(lim, 0);
            if (bas <= cof.size())
                A.cof.assign(cof.begin(), cof.begin() + bas);
            else
                A.cof = cof;
            A.cof.resize(lim, 0);
            std::vector<int> rev(generateRev(lim, limpow));
            A.NTT(rev, lim, 0);
            B.NTT(rev, lim, 0);
            for (auto i : range(lim))
                B.cof[i] = (LL)B.cof[i] * (2 + mod - (LL)B.cof[i] * A.cof[i] % mod) % mod;
            B.NTT(rev, lim, 1);
            std::fill(B.cof.begin() + bas, B.cof.end(), 0);
            bas <<= 1;
            lim <<= 1;
            ++limpow;
        }
        B.cof.resize(siz);
    }
    /* 两次MTT的任意模数多项式求逆，1e5 O2 550ms，无O2 2.11s */
    void F_inv(int siz, Polynomial &B)
    {
        if (siz == 1)
        {
            B.cof.emplace_back(inv(LL(round(cof[0].real())), mod));
            return;
        }
        F_inv((siz + 1) >> 1, B);
        Polynomial C;
        C.cof.assign(cof.begin(), cof.begin() + siz);
        Polynomial BC(MTT_FFT(B, C));
        for (auto &i : BC.cof)
            i = LL(round(i.real())) % mod;
        Polynomial BBC(MTT_FFT(BC, B));
        B.cof.resize(siz, 0);
        for (int i = 0; i < siz; ++i)
        {
            B.cof[i] = msub(
                madd(
                    LL(round(B.cof[i].real())),
                    LL(round(B.cof[i].real()))),
                LL(round(BBC.cof[i].real())) % mod);
        }
    }
    /* G2 = (G1^2 + A)/2G1 */
    Polynomial getsqrt()
    {
        Polynomial B;
        int siz = cof.size();
        LL s1, s2;
        Cipolla<LL>::solve((LL)cof[0], (LL)mod, s1, s2);
        if (s2 < s1)
            swap(s2, s1);
        B.cof.emplace_back(s1);
        LL bas = 2, lim = 4, limpow = 2;
        Polynomial A;
        // T inv2 = inv(2, mod);
        while (bas < (siz << 1))
        {
            Polynomial Binv(B.getinv(bas));
            B.cof.resize(lim, 0);
            if (bas <= cof.size())
                A.cof.assign(cof.begin(), cof.begin() + bas);
            else
                A.cof = cof;
            A.cof.resize(lim, 0);
            std::vector<int> rev(generateRev(lim, limpow));
            Binv.cof.resize(lim);
            A.NTT(rev, lim, 0);
            B.NTT(rev, lim, 0);
            Binv.NTT(rev, lim, 0);

            for (auto i : range(lim))
            {
                B.cof[i] = (LL)Binv.cof[i] * (A.cof[i] + (LL)B.cof[i] * B.cof[i] % mod) % mod;
                B.cof[i] = (B.cof[i] & 1) ? (B.cof[i] + mod >> 1) : B.cof[i] >> 1;
            }

            B.NTT(rev, lim, 1);
            std::fill(B.cof.begin() + bas, B.cof.end(), 0);
            bas <<= 1;
            lim <<= 1;
            ++limpow;
        }
        B.cof.resize(siz);
        return B;
    }

    /* siz为需要求的多项式逆的次数，为0时默认取自己次数的 */
    Polynomial getinv(int siz = 0)
    {
        if (!siz)
            siz = cof.size();
        int orisiz = cof.size();
        cof.resize(siz);
        Polynomial B;
        // LL lim, limpow, retsize;
        // Resize(*this, *this, lim, limpow, retsize);
        N_inv(siz, B); // N_inv为使用NTT，F_inv为使用MTT
        B.cof.resize(siz);
        cof.resize(orisiz);
        return B;
    }

    Polynomial operator*(const Polynomial &rhs)
    {
        return NTTMul(*this, rhs);
    }
    /* 获取F(x) = G(x) * Q(x) +  R(x)的Q(x) */
    Polynomial operator/(Polynomial G)
    {
        Polynomial F(*this);
        int beforen = F.cof.size();
        std::reverse(F.cof.begin(), F.cof.end());
        std::reverse(G.cof.begin(), G.cof.end());
        int beforem = G.cof.size();
        F.cof.resize(beforen - beforem + 1);
        Polynomial tmp(F * G.getinv(beforen));
        // G.cof.resize(beforem);
        tmp.cof.resize(beforen - beforem + 1);
        std::reverse(tmp.cof.begin(), tmp.cof.end());
        // std::reverse(cof.begin(), cof.end());
        return tmp;
    }

    /* 获取F(x) = G(x) * Q(x) +  R(x)的R(x) */
    static Polynomial getremain(Polynomial &F, Polynomial &G, Polynomial &Q)
    {
        Polynomial C(G * Q);
        C.cof.resize(G.cof.size() - 1);
        for (auto i : range(G.cof.size() - 1))
            C.cof[i] = F.msub(F.cof[i], C.cof[i]);
        return C;
    }

    static std::vector<int> generateRev(LL lim, LL limpow)
    {
        std::vector<int> rev(lim, 0);
        for (auto i : range(1, lim))
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (limpow - 1));
        return rev;
    }

    static std::vector<T> generateWn(LL lim)
    {
        std::vector<T> Wn;
        for (int i = 0; i < lim; i++)
            Wn.emplace_back(cos(M_PI / lim * i), sin(M_PI / lim * i));
        return Wn;
    }

    /* NTT卷积 板题4.72s */
    static Polynomial NTTMul(Polynomial A, Polynomial B)
    {
        LL lim, limpow, retsiz;
        Resize(A, B, lim, limpow, retsiz);

        std::vector<int> rev(generateRev(lim, limpow));
        A.NTT(rev, lim, 0);
        B.NTT(rev, lim, 0);
        for (auto i : range(lim))
            A.cof[i] = (A.cof[i] * B.cof[i] % A.mod);
        A.NTT(rev, lim, 1);

        A.cof.resize(retsiz - 1);

        return A;
    }
    /* FFT卷积 板题1.98s 使用手写复数 -> 1.33s*/
    static Polynomial FFTMul(Polynomial A, Polynomial B)
    {
        LL lim, limpow, retsiz;
        Resize(A, B, lim, limpow, retsiz);

        std::vector<int> rev(generateRev(lim, limpow));
        std::vector<T> Wn(generateWn(lim));
        A.FFT(rev, lim, 0, Wn);
        B.FFT(rev, lim, 0, Wn);
        for (auto i : range(lim))
            A.cof[i] *= B.cof[i];
        A.FFT(rev, lim, 1, Wn);

        A.cof.resize(retsiz - 1);
        return A;
    }

    inline static void Resize(Polynomial &A, Polynomial &B, LL &lim, LL &limpow, LL &retsiz)
    {
        lim = 1;
        limpow = 0;
        retsiz = A.cof.size() + B.cof.size();
        while (lim <= retsiz)
            lim <<= 1, ++limpow;
        A.cof.resize(lim, 0);
        B.cof.resize(lim, 0);
    }

    static Polynomial MTT_FFT(const Polynomial &A, const Polynomial &B)
    {
        LL lim, limpow, retsiz;
        Polynomial A0, B0;
        LL thr = sqrt(A.mod) + 1; // 拆系数阈值
        for (auto i : A.cof)
        {
            LL tmp = i.real();
            A0.cof.emplace_back(tmp / thr, tmp % thr);
        }
        for (auto i : B.cof)
        {
            LL tmp = i.real();
            B0.cof.emplace_back(tmp / thr, tmp % thr);
        }
        Resize(A0, B0, lim, limpow, retsiz);

        std::vector<int> rev(generateRev(lim, limpow));
        std::vector<T> Wn(generateWn(lim));

        A0.FFT(rev, lim, 0, Wn);
        B0.FFT(rev, lim, 0, Wn);
        std::vector<T> Acp(A0.cof);
        std::vector<T> Bcp(B0.cof);
        const T IV(0, 1);
        const T half(0.5);
        for (int ii = 0; ii < lim; ++ii)
        {
            T i = A0.cof[ii];
            T j = (Acp[ii ? lim - ii : 0]).conj();
            T a0 = (j + i) * half;
            T a1 = (j - i) * half * IV;
            i = B0.cof[ii];
            j = (Bcp[ii ? lim - ii : 0]).conj();
            T b0 = (j + i) * half;
            T b1 = (j - i) * half * IV;
            A0.cof[ii] = a0 * b0 + IV * a1 * b0;
            B0.cof[ii] = a0 * b1 + IV * a1 * b1;
        }
        A0.FFT(rev, lim, 1, Wn);
        B0.FFT(rev, lim, 1, Wn);

        for (int i = 0; i < retsiz - 1; ++i)
        {
            T &ac = A0.cof[i];
            T &bc = B0.cof[i];
            A0.cof[i] = thr * thr * (__int128)round(ac.real()) % A.mod +
                        thr * (__int128)round(ac.imag() + bc.real()) % A.mod +
                        (__int128)round(bc.imag()) % A.mod;
        }
        A0.cof.resize(retsiz - 1);
        return A0;
    }
};
/* 使用手写的以后 2.00s -> 1.33s*/
template <typename T>
struct Complex
{
    T re_al, im_ag;
    inline T &real() { return re_al; }
    inline T &imag() { return im_ag; }
    Complex() { re_al = im_ag = 0; }
    Complex(T x) : re_al(x), im_ag(0) {}
    Complex(T x, T y) : re_al(x), im_ag(y) {}
    inline Complex conj() { return Complex(re_al, -im_ag); }
    inline Complex operator+(Complex rhs) const { return Complex(re_al + rhs.re_al, im_ag + rhs.im_ag); }
    inline Complex operator-(Complex rhs) const { return Complex(re_al - rhs.re_al, im_ag - rhs.im_ag); }
    inline Complex operator*(Complex rhs) const { return Complex(re_al * rhs.re_al - im_ag * rhs.im_ag,
                                                                 im_ag * rhs.re_al + re_al * rhs.im_ag); }
    inline Complex operator*=(Complex rhs) { return (*this) = (*this) * rhs; }
    //(a+bi)(c+di) = (ac-bd) + (bc+ad)i
    friend inline Complex operator*(T x, Complex cp) { return Complex(x * cp.re_al, x * cp.im_ag); }
    inline Complex operator/(T x) const { return Complex(re_al / x, im_ag / x); }
    inline Complex operator/=(T x) { return (*this) = (*this) / x; }
    friend inline Complex operator/(T x, Complex cp) { return x * cp.conj() / (cp.re_al * cp.re_al - cp.im_ag * cp.im_ag); }
    inline Complex operator/(Complex rhs) const
    {
        return (*this) * rhs.conj() / (rhs.re_al * rhs.re_al - rhs.im_ag * rhs.im_ag);
    }
    inline Complex operator/=(Complex rhs) { return (*this) = (*this) / rhs; }
    inline Complex operator=(T x)
    {
        this->im_ag = 0;
        this->re_al = x;
        return *this;
    }
    inline T length() { return sqrt(re_al * re_al + im_ag * im_ag); }
};
using _MTT = Complex<double>;
using _NTT = long long;