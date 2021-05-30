# 数论

## 欧拉筛

```cpp
// #define ORAFM 2333
int prime[ORAFM + 5], prime_number = 0, prv[ORAFM + 5];
// 莫比乌斯函数
int mobius[ORAFM + 5];
// 欧拉函数
LL phi[ORAFM + 5];

bool marked[ORAFM + 5];

void ORAfliter(LL MX)
{
    mobius[1] = phi[1] = 1;
    for (unsigned int i = 2; i <= MX; i++)
    {
        if (!marked[i])
        {
            prime[++prime_number] = i;
            prv[i] = i;
            phi[i] = i - 1;
            mobius[i] = -1;
        }
        for (unsigned int j = 1; j <= prime_number && i * prime[j] <= MX; j++)
        {
            marked[i * prime[j]] = true;
            prv[i * prime[j]] = prime[j];
            if (i % prime[j] == 0)
            {
                phi[i * prime[j]] = prime[j] * phi[i];
                break;
            }
            phi[i * prime[j]] = phi[prime[j]] * phi[i];
            mobius[i * prime[j]] = -mobius[i]; // 平方因数不会被处理到，默认是0
        }
    }
    // 这句话是做莫比乌斯函数和欧拉函数的前缀和
    for (unsigned int i = 2; i <= MX; ++i)
    {
        mobius[i] += mobius[i - 1];
        phi[i] += phi[i - 1];
    }
}
```

## min_25筛框架
```cpp
inline void prework(LL n)
{
    int tot = 0;
    for (LL l = 1, r; l <= n; l = r + 1)
    {
        r = n / (n / l); // 数论分块？
        w[++tot] = n / l;
        // g1[tot] = w[tot] % mo;
        // g2[tot] = (g1[tot] * (g1[tot] + 1) >> 1) % mo * ((g1[tot] << 1) + 1) % mo * inv3 % mo;
        // g2[tot]--;
        // g1[tot] = (g1[tot] * (g1[tot] + 1) >> 1) % mo - 1;
        valposition(n / l, n) = tot;
        g1[tot] = n / l - 1;
        // g2[tot] = n / l - 1;
    }
    for (int i = 1; i <= prime_number; i++)
    {
        for (int j = 1; j <= tot and (LL) prime[i] * prime[i] <= w[j]; j++)
        {
            LL n_div_m_val = w[j] / prime[i];
            if (n_div_m_val)
            {
                int n_div_m = valposition(n_div_m_val, n); // m: prime[i]
                g1[j] -= g1[n_div_m] - (i - 1);            // 枚举第i个质数，所以可以直接减去i-1，这里无需记录sp
            }
            // g1[j] -= (LL)prime[i] * (g1[k] - sp1[i - 1] + mo) % mo;
            // g2[j] -= (LL)prime[i] * prime[i] % mo * (g2[k] - sp2[i - 1] + mo) % mo;
            // g1[j] %= mo,
            //     g2[j] %= mo;
            // if (g1[j] < 0)
            //     g1[j] += mo;
            // if (g2[j] < 0)
            //     g2[j] += mo;
        }
    }
}
// 1~x中最小质因子大于y的函数值
inline LL S_(LL x, int y)
{
    if (prime[y] >= x)
        return 0;
    int k = valposition(x, n);
    // 此处g1、g2代表1、2次项
    LL ans = (g2[k] - g1[k] + mo - (sp2[y] - sp1[y]) + mo) % mo;
    // ans = (ans + mo) % mo;
    for (int i = y + 1; i <= prime_number and prime[i] * prime[i] <= x; ++i)
    {
        LL pe = prime[i];
        for (int e = 1; pe <= x; e++, pe *= prime[i])
        {
            LL xx = pe % mo;
            // 大概这里改ans？原题求p^k*(p^k-1)
            ans = (ans + xx * (xx - 1) % mo * (S_(x / pe, i) + (e != 1))) % mo;
        }
    }
    return ans % mo;
}

```


## 卢卡斯定理

```cpp
LL fact[LUCASM];

inline void get_fact(LL fact[], LL length, LL mo) // 预处理阶乘
{
    fact[0] = 1;
    fact[1] = 1;
    for (auto i = 2; i < length; i++)
        fact[i] = fact[i - 1] * i % mo;
}
// 需要先预处理出fact[]，即阶乘
inline LL C(LL m, LL n, LL p)
{
    return m < n ? 0 : fact[m] * inv(fact[n], p) % p * inv(fact[m - n], p) % p;
}
inline LL lucas(LL m, LL n, LL p) // 求解大数组合数C(m,n) % p,传入依次是下面那个m和上面那个n和模数p（得是质数
{
    return n == 0 ? 1 % p : lucas(m / p, n / p, p) * C(m % p, n % p, p) % p;
}
```

## EXCRT

```cpp
inline LL EXCRT(LL factors[], LL remains[], LL length) // 传入除数表，剩余表和两表的长度，若没有解，返回-1，否则返回合适的最小解
{
    bool valid = true;
    for (auto i = 1; i < length; i++)
    {
        LL GCD = gcd(factors[i], factors[i - 1]);
        LL M1 = factors[i];
        LL M2 = factors[i - 1];
        LL C1 = remains[i];
        LL C2 = remains[i - 1];
        LL LCM = M1 * M2 / GCD;
        if ((C1 - C2) % GCD != 0)
        {
            valid = false;
            break;
        }
        factors[i] = LCM;
        remains[i] = (inv(M2 / GCD, M1 / GCD) * (C1 - C2) / GCD) % (M1 / GCD) * M2 + C2; // 对应合并公式
        remains[i] = (remains[i] % factors[i] + factors[i]) % factors[i];                // 转正
    }
    return valid ? remains[length - 1] : -1;
}
```

## 扩欧求逆元

```cpp
inline void exgcd(LL a, LL b, LL &x, LL &y)
{
    if (!b)
    {
        x = 1;
        y = 0;
        return;
    }
    exgcd(b, a % b, y, x);
    y -= a / b * x;
}
inline LL inv(LL a, LL mo)
{
    LL x, y;
    exgcd(a, mo, x, y);
    return x >= 0 ? x : x + mo;
}
```

## FFT

```cpp
struct Complex
{
    double re, im;
    Complex(double _re, double _im)
    {
        re = _re;
        im = _im;
    }
    Complex(double _re)
    {
        re = _re;
        im = 0;
    }
    Complex() {}
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
```

## 拉格朗日插值

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
const int maxn = 2010;
using ll = long long;
ll mod = 998244353;
ll n, k, x[maxn], y[maxn], ans, s1, s2;
ll powmod(ll a, ll x) {
	ll ret = 1ll, nww = a;
	while (x) {
		if (x & 1) ret = ret * nww % mod;
			nww = nww * nww % mod;
		x >>= 1;
	}
	return ret;
}
ll inv(ll x) { return powmod(x, mod - 2); }
int main() {
	scanf("%lld%lld", &n, &k);
	for (int i = 1; i <= n; i++) scanf("%lld%lld", x + i, y + i);
	for (int i = 1; i <= n; i++) {
		s1 = y[i] % mod;
		s2 = 1ll;
		for (int j = 1; j <= n; j++)
			if (i != j)
				s1 = s1 * (k - x[j]) % mod, s2 = s2 * ((x[i] - x[j] % mod) % mod) % mod;
		ans += s1 * inv(s2) % mod;
		ans = (ans + mod) % mod;
	}
	printf("%lld\n", ans);
	return 0;
}
```

## 高斯消元

```cpp
void Gauss() {
	for(int i = 0; i < n; i ++) {
		r = i;
		for(int j = i + 1; j < n; j ++)
			if(fabs(A[j][i]) > fabs(A[r][i])) r = j;
		if(r != i) for(int j = 0; j <= n; j ++) std :: swap(A[r][j], A[i][j]);

		for(int j = n; j >= i; j --) {
			for(int k = i + 1; k < n; k ++)
				A[k][j] -= A[k][i] / A[i][i] * A[i][j];
		}
	}

	for(int i = n - 1; i >= 0; i --) {
		for(int j = i + 1; j < n; j ++)
			A[i][n] -= A[j][n] * A[i][j];
		A[i][n] /= A[i][i];
	}
}
```

## 公式

卡特兰数 K(x) = C(2*x, x) / (x + 1)

## 来自bot的球盒问题

```py
def A072233_list(n: int, m: int, mod=0) -> list:
    """n个无差别球塞进m个无差别盒子方案数"""
    mod = int(mod)
    f = [[0] * (m + 1)] * (n + 1)
    f[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, min(i+1, m+1)): # 只是求到m了话没必要打更大的
            f[i][j] = f[i-1][j-1] + f[i-j][j]
            if mod: f[i][j] %= mod
    return f

def A048993_list(n: int, m: int, mod=0) -> list:
    """第二类斯特林数"""
    mod = int(mod)
    f = [1] + [0] * m
    for i in range(1, n+1):
        for j in range(min(m, i), 0, -1):
            f[j] = f[j-1] + f[j] * j
            if mod: f[j] %= mod
        f[0] = 0
    return f


def A000110_list(m, mod=0):
    """集合划分方案总和，或者叫贝尔数"""
    mod = int(mod)
    A = [0 for i in range(m)]
    # m -= 1
    A[0] = 1
    # R = [1, 1]
    for n in range(1, m):
        A[n] = A[0]
        for k in range(n, 0, -1):
            A[k-1] += A[k]
            if mod: A[k-1] %= mod
        # R.append(A[0])
    # return R
    return A[0]

async def 球盒(*attrs, kwargs={}):
    """求解把n个球放进m个盒子里面有多少种方案的问题。
必须指定盒子和球以及允不允许为空三个属性。
用法：
    #球盒 <盒子相同？(0/1)><球相同？(0/1)><允许空盒子？(0/1)> n m
用例：
    #球盒 110 20 5
    上述命令求的是盒子相同，球相同，不允许空盒子的情况下将20个球放入5个盒子的方案数。"""
    # 参考https://www.cnblogs.com/sdfzsyq/p/9838857.html的算法
    if len(attrs)!=3:
        return '不是这么用的！请输入#h #球盒'
    n, m = map(int, attrs[1:3])
    if attrs[0] == '110':
        f = A072233_list(n, m)
        return f[n][m]
    elif attrs[0] == '111':
        f = A072233_list(n, m)
        return sum(f[-1])
    elif attrs[0] == '100':
        return A048993_list(n, m)[-1]
    elif attrs[0] == '101':
        return sum(A048993_list(n, m))
    elif attrs[0] == '010':
        return comb(n-1, m-1)
    elif attrs[0] == '011':
        return comb(n+m-1, m-1)
    elif attrs[0] == '000': # 求两个集合的满射函数的个数可以用
        return A048993_list(n, m)[-1] * math.factorial(m)
    elif attrs[0] == '001':
        return m**n

```