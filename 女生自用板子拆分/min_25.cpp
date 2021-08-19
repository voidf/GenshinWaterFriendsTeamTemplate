#include "Headers.cpp"
// min_25筛，需要先欧拉筛打表n**0.5 => P1835
int prime[ORAFM + 5], prime_number = 0;

LL sp1[ORAFM + 5], sp2[ORAFM + 5];
LL ind1[ORAFM + 5], ind2[ORAFM + 5];

LL sqr;

LL &valposition(LL x, LL n) { return x <= sqr ? ind1[x] : ind2[n / x]; }

LL g1[ORAFM + 5], g2[ORAFM + 5];
LL w[ORAFM + 5];
bool marked[ORAFM + 5];

const LL inv3 = 333333336; // 3 / 1e9+7

inline void ORAfliter(LL MX)
{
    marked[1] = 1;
    for (unsigned int i = 2; i <= MX; i++)
    {
        if (!marked[i])
        {
            prime[++prime_number] = i;
            // sp1[prime_number] = (sp1[prime_number - 1] + i) % mo; // sp:记录质数的答案，即f(m-1, (m-1)**0.5)
            // sp2[prime_number] = (sp2[prime_number - 1] + (LL)i * i) % mo;
        }
        for (unsigned int j = 1; j <= prime_number && i * prime[j] <= MX; j++)
        {
            marked[i * prime[j]] = true;
            if (i % prime[j] == 0)
            {
                break;
            }
        }
    }
}
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


// 递归，分段缓存版本
unordered_map<ULL, LL> UM;
unordered_map<unsigned, unsigned> IM;

LL ans[100010];
vector<vector<pair<LL, LL>>> QUERY(17, vector<pair<LL, LL>>());

unsigned gfi(unsigned n, int j)
{
    unsigned mpk = unsigned(j) * 1000000001 + n;
    if (IM.count(mpk))
        return IM[mpk];
    else
    {
        LL ret;
        if (n < 2)
            ret = 0;
        else if (n == 2)
            ret = 1;
        else if (j < 1)
            ret = n - 1;
        else if (prime[j] * prime[j] > n)
            ret = gfi(n, j - 1);
        else
            ret = gfi(n, j - 1) - (gfi(n / prime[j], j - 1) - (j - 1));
        // if (n < 1e9)
        //     return UM[mpk] = ret;
        return ret;
    }
}

LL gf(LL n, LL j)
{
    if (n < 1e9)
        return gfi(unsigned(n), j);
    ULL mpk = ULL(j) * 1000000000000000001 + n;
    if (UM.count(mpk))
        return UM[mpk];
    else
    {
        LL ret;
        if (n < 2)
            ret = 0;
        else if (n == 2)
            ret = 1;
        else if (j < 1)
            ret = n - 1;
        else if (prime[j] * prime[j] > n)
            ret = gf(n, j - 1);
        else
        {
            // ret = gf(n, j - 1) -  (gf(n / prime[j], j - 1)) - (j - 1);
            ret = gf(n, j - 1);
            LL dv = n / prime[j];
            LL ret2 = (n < 1e9 ? gfi(dv, j - 1) : gf(dv, j - 1)) - (j - 1);
            ret -= ret2;
        }
        // if (n < 1e9)
        //     return UM[mpk] = ret;
        return UM[mpk] = ret;
    }
}

// min_25结束