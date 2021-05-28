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
