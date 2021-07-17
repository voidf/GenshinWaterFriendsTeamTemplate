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
        LL xMin = x1 + xlower * dx;            // x的最小正整数解
        LL yMin = y1 - yupper * dy;            // y的最小正整数解
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