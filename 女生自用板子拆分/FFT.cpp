#include "Headers.cpp"

struct Complex
{
    double re, im;
    Complex(double _re, double _im) : re(_re), im(_im) {}
    Complex(double _re) : re(_re), im(0) {}
    Complex() {}
    inline double real() { return re; }
    inline double imag() { return im; }
};
Complex operator+(const Complex &c1, const Complex &c2) { return Complex(c1.re + c2.re, c1.im + c2.im); }
Complex operator-(const Complex &c1, const Complex &c2) { return Complex(c1.re - c2.re, c1.im - c2.im); }
Complex operator*(const Complex &c1, const Complex &c2) { return Complex(c1.re * c2.re - c1.im * c2.im, c1.re * c2.im + c1.im * c2.re); }

void FFT(complex<double> a[], LL n, LL inv, LL rev[]) // FFT系列没有外部依赖数组，不用打ifdef
{
    for (auto i = 0; i < n; i++)
        if (i < rev[i])
            swap(a[i], a[rev[i]]);
    for (auto mid = 1; mid < n; mid <<= 1)
    {
        complex<double> tmp(cos(pi / mid), inv * sin(pi / mid));
        for (auto i = 0; i < n; i += mid << 1)
        {
            complex<double> omega(1, 0);
            for (auto j = 0; j < mid; j++, omega *= tmp)
            {
                auto x = a[i + j], y = omega * a[i + j + mid];
                a[i + j] = x + y, a[i + j + mid] = x - y;
            }
        }
    }
}

LL *FFTArrayMul(LL A[], LL B[], LL Alen, LL Blen)
{
    LL max_length = max(Alen, Blen);
    LL limit = 1;
    LL bit = 0;
    while (limit < max_length << 1)
        bit++, limit <<= 1;

    auto rev = new LL[limit + 5];
    mem(rev, 0);

    for (auto i = 0; i <= limit; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));

    // auto a = new complex<double>[limit + 5];
    // auto b = new complex<double>[limit + 5];
    complex<double> a[limit + 5], b[limit + 5];

    // for (auto i = 0; i < limit; i++)
    // {
    //     a[i] = i >= Alen ? 0 : A[Alen - i - 1] ^ 0x30;
    //     b[i] = i >= Blen ? 0 : B[Blen - i - 1] ^ 0x30;
    // } // 右对齐的输入方式，类似下面的大数乘法板子，答案需要去除前导零
    for (auto i = 0; i < max_length; i++) // 左对齐，段错误的坑，得用max_length
    {
        a[i] = A[i];
        b[i] = B[i];
    }
    static LL *c = new LL[limit + 5];
    mem(c, 0);

    FFT(a, limit, 1, rev);
    FFT(b, limit, 1, rev);

    for (auto i = 0; i <= limit; i++)
        a[i] *= b[i];

    FFT(a, limit, -1, rev); // 1是FFT变化，-1是逆变换
    for (auto i = 0; i <= limit; i++)
        c[i] = (LL)(a[i].real() / limit + 0.5); // +0.5即四舍五入
    return c;                                   // 左对齐多项式卷积结果有效位数为n+m-1
}

string FFTBigNumMul(string &A, string &B)
{

    auto max_length = max(A.length(), B.length());

    LL limit = 1;
    LL bit = 0;

    while (limit < max_length << 1)
        bit++, limit <<= 1;

    LL rev[limit + 5] = {0};

    for (auto i = 0; i <= limit; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));

    complex<double> a[limit + 5], b[limit + 5];

    for (auto i = 0; i < limit; i++)
    {
        a[i] = i >= A.length() ? 0 : A[A.length() - i - 1] ^ 0x30;
        b[i] = i >= B.length() ? 0 : B[B.length() - i - 1] ^ 0x30;
    }

    LL c[limit + 5] = {0};

    FFT(a, limit, 1, rev);
    FFT(b, limit, 1, rev);
    for (auto i = 0; i <= limit; i++)
        a[i] *= b[i];
    FFT(a, limit, -1, rev); // 1是FFT变化，-1是逆变换

    for (auto i = 0; i <= limit; i++)
        c[i] = (LL)(a[i].real() / limit + 0.5); // +0.5即四舍五入
    bool zerofliter = false;
    for (auto i = 0; i < limit; i++)
    {
        c[i + 1] += c[i] / 10;
        c[i] %= 10;
    }
    char output[limit + 5] = {0};
    LL outputPtr = 0;
    // mem(output, 0);
    for (auto i = limit; i >= 0; i--)
    {
        if (c[i] == 0 and zerofliter == 0) // 去前导零
            continue;
        zerofliter = true;
        output[outputPtr++] = c[i] ^ 0x30;
    }
    string res(output);
    if (!res.length())
        res = string("0");
    return res;
}