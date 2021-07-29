#include "Headers.cpp"

inline LL modadd(LL &x, LL y) { return (x += y) >= mo ? x -= mo : x; }
inline LL madd(LL x, LL y) { return (x += y) >= mo ? x - mo : x; }
inline LL modsub(LL &x, LL y) { return (x -= y) < 0 ? x += mo : x; }
inline LL msub(LL x, LL y) { return (x -= y) < 0 ? x + mo : x; }

template <typename IntegerType>
inline IntegerType power(IntegerType a, IntegerType n)
{
    IntegerType res = 1;
    while (n)
    {
        if (n & 1)
            res = res * a;
        a = a * a;
        n >>= 1;
    }
    return res;
}

template <class T>
T gcd(T a, T b) { return !b ? a : gcd(b, a % b); }

inline LL exgcd(LL a, LL b, LL &x, LL &y)
{
    if (!b)
    {
        x = 1;
        y = 0;
        return a;
    }
    LL res = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return res;
}

inline LL inv(LL a, LL mo)
{
    LL x, y;
    exgcd(a, mo, x, y);
    return x >= 0 ? x : x + mo; // 为0时无解
}

//递推求法
std::vector<LL> getInvRecursion(LL upp, LL mod)
{
    std::vector<LL> vinv(1, 0);
    vinv.emplace_back(1);
    for (LL i = 2; i <= upp; i++)
        vinv.emplace_back((mod - mod / i) * vinv[mod % i] % mod);
    return vinv;
}

/* 解同余方程ax + by = c */
void exgcd_solve()
{
    qr(a);
    qr(b);
    qr(c);

    LL GCD = exgcd(a, b, x, y);
    if (c % GCD != 0) // 无解
    {
        puts("-1");
        return;
    }

    LL xishu = c / GCD;

    LL x1 = x * xishu;
    LL y1 = y * xishu;
    // 为了满足 a * (x1 + db) + b * (y1 - da) = c的形式
    // x1, y1 是特解，通过枚举【实数】d可以得到通解
    LL dx = b / GCD; // 构造 x = x1 + s * dx ，即a的系数
    LL dy = a / GCD; // 构造 y = y1 - s * dy ，即b的系数
                     // 这步的s就可以是整数了
    // 限制 x>0 => x1 + s * dx > 0 => s > - x1 / dx (实数)
    // 限制 y>0 => y1 - s * dy > 0 => s < y1 / dy (实数)

    LL xlower = ceil(double(-x1 + 1) / dx); // s可能的最小值
    LL yupper = floor(double(y1 - 1) / dy); // s可能的最大值
    if (xlower > yupper)
    {
        LL xMin = x1 + xlower * dx; // x的最小正整数解
        LL yMin = y1 - yupper * dy; // y的最小正整数解
        printf("%lld %lld\n", xMin, yMin);
    }
    else
    {
        LL s_range = yupper - xlower + 1; // 正整数解个数
        LL xMax = x1 + yupper * dx;       // x的最大正整数解
        LL xMin = x1 + xlower * dx;       // x的最小正整数解
        LL yMax = y1 - xlower * dy;       // y的最大正整数解
        LL yMin = y1 - yupper * dy;       // y的最小正整数解
        printf("%lld %lld %lld %lld %lld\n", s_range, xMin, yMin, xMax, yMax);
    }
}

/* 取 l <= dx%m <= r 的最小非负x */
LL modinq(LL m, LL d, LL l, LL r)
{
    // 0 <= l <= r < m, d < m, minimal non-negative solution
    if (r < l)
        return -1;
    if (l == 0)
        return 0;
    if (d == 0)
        return -1;
    if ((r / d) * d >= l)
        return (l - 1) / d + 1;
    LL res = modinq(d, m % d, (d - r % d) % d, (d - l % d) % d);
    return res == -1 ? -1 : (m * res + l - 1) / d + 1;
}

/* 求1~x的和 */
LL pref(LL x) { return (x) * (x + 1) >> 1; }

/* 求 ax%p x从1到p-1 的序列的逆序对数 */
LL calcinvs(LL a, LL p)
{
    a %= p;
    if (a * 2 > p)
    {
        return pref(p - 2) - calcinvs(p - a, p);
    }

    LL G = gcd(a, p);
    a /= G;
    p /= G;
    LL orires;

    if (a <= 1 or a == p)
    {
        orires = 0;
    }
    else if (a == p - 1)
    {
        orires = pref(a - 1);
    }
    else
    {
        LL kuaidaxiao = p / a;
        LL yushu = p % a;

        LL pf = pref(kuaidaxiao);
        LL pf2 = pref(a - 1);

        LL offset1 = pf * pf2 & 1;
        if (yushu == 0)
        {
            orires = offset1;
        }
        else
        {
            LL step = a - yushu;

            LL gouctr = yushu - 1;

            LL shengxia = a - gouctr;

            LL atoP = shengxia - 1;

            LL firstarr = modinq(a, step, yushu + 1, a - 1) * step % a - yushu;

            LL offset2 = (kuaidaxiao & 1) ? (pref(atoP - 2) - calcinvs(firstarr, atoP) & 1) : (calcinvs(step % yushu, yushu) & 1);
            orires = offset1 ^ offset2;
            // orires = offset1 + offset2;
        }
    }
    // return ;
    // LL orires = pf * pf2 + (kuaidaxiao + 1) * calcinvs(step % yushu, yushu) - (kuaidaxiao) * (pref(atoP - 2) - calcinvs(firstarr, atoP));
    orires *= G;
    return 1 & orires + pref(G - 1) * pref(p - 1);
    // return pf * pf2 + (kuaidaxiao + 1) * calcinvs(step % yushu, yushu) - (kuaidaxiao) * (pref(atoP - 2) - calcinvs(firstarr, atoP));
}