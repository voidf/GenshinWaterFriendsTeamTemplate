#define N 2048
typedef bitset<N> Bint;
bool operator<(const Bint &a, const Bint &b)
{
    for (int i = a.size() - 1; i >= 0; --i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return 0;
}
bool operator>(const Bint &a, const Bint &b)
{
    for (int i = a.size() - 1; i >= 0; --i)
        if (a[i] != b[i])
            return a[i] > b[i];
    return 0;
}
bool operator<=(const Bint &a, const Bint &b) { return !(a > b); }
bool operator>=(const Bint &a, const Bint &b) { return !(a < b); }
Bint operator+(const Bint &a, const Bint &b) { return b.any() ? (a ^ b) + ((a & b) << 1) : a; }
Bint &operator+=(Bint &a, const Bint &b) { return a = a + b; }
Bint operator-(const Bint &a) { return Bint(1) + ~a; }
//Bint operator -(const Bint &a,const Bint &b){return a+(-b);}
Bint operator-(const Bint &a, const Bint &b) { return b.any() ? (a ^ b) - ((~a & b) << 1) : a; }
Bint &operator-=(Bint &a, const Bint &b) { return a = a - b; }
Bint operator*(Bint a, Bint b)
{
    Bint r(0);
    for (; b.any(); b >>= 1, a <<= 1)
        if (b[0])
            r += a;
    return r;
}
Bint &operator*=(Bint &a, const Bint &b) { return a = a * b; }
pair<Bint, Bint> divide(Bint a, const Bint &b)
{
    Bint c = 0;
    int i = 0;
    while (b << (i + 1) <= a)
        ++i;
    for (; i >= 0; --i)
        if (a >= (b << i))
            a -= b << i, c.set(i, 1);
    return make_pair(c, a);
}
Bint operator/(const Bint &a, const Bint &b) { return divide(a, b).first; }
Bint &operator/=(Bint &a, const Bint &b) { return a = a / b; }
Bint operator%(const Bint &a, const Bint &b) { return divide(a, b).second; }
Bint &operator%=(Bint &a, const Bint &b) { return a = a % b; }
inline void read(Bint &x)
{
    char ch;
    int bo = 0;
    x = 0;
    for (ch = getchar(); ch < '0' || ch > '9'; ch = getchar())
        if (ch == '-')
            bo = 1;
    for (; ch >= '0' && ch <= '9'; x = (x << 3) + (x << 1) + (ch - '0'), ch = getchar())
        ;
    if (bo)
        x = -x;
}

inline void printBint(Bint x)
{
    vector<Bint> v;
    if (x == 0)
        printf("0");
    for (Bint y = 1; y <= x; y *= 10)
        v.push_back(y);
    for (int i = v.size() - 1; i >= 0; --i)
    {
        int t = 0;
        while (x >= (v[i] << 2))
            x -= v[i] << 2, t += 4;
        while (x >= (v[i] << 1))
            x -= v[i] << 1, t += 2;
        while (x >= v[i])
            x -= v[i], ++t;
        printf("%d", t);
    }
    printf("\n");
}


