// v2021.5.22 主席树更面向对象
#pragma GCC optimize(1)
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#pragma GCC target("avx")
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-fwhole-program")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-funsafe-loop-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
// __builtin_popcount(); // 数1
// __builtin_clz(); // 数前导零
// __builtin_ctz(); // 数后导零

#define in ,
#define foreach(...) foreach_ex(foreach_in, (__VA_ARGS__))
#define foreach_ex(m, wrapped_args) m wrapped_args
#define foreach_in(e, a) for (int i = 0, elem *e = a->elems; i != a->size; i++, e++)

#define sign(_x) (_x < 0)
#define range_4(__iter__, __from__, __to__, __step__) for (LL __iter__ = __from__; __iter__ != __to__ && sign(__to__ - __from__) == sign(__step__); __iter__ += __step__)
#define range_3(__iter__, __from__, __to__) range_4(__iter__, __from__, __to__, 1)
#define range_2(__iter__, __to__) range_4(__iter__, 0, __to__, 1)
#define range_1(__iter__, __to__) range_4(__iter__, 0, 1, 1)
#define get_range(_1, _2, _3, _4, _Func, ...) _Func
#define range(...) get_range(__VA_ARGS__, range_4, range_3, range_2, range_1, ...)(__VA_ARGS__)

#include <ext/pb_ds/tag_and_trait.hpp>
#define _CRT_SECURE_NO_WARNINGS
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma comment(linker, "/stack:200000000")

// 玄学优化，没用的，别想了
#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
// #include <bits/extc++.h>
using namespace std;
typedef long long LL;
// typedef __int128 LL;
typedef unsigned long long ULL;

#define pi acos(-1)
#define M 200010
#define endl '\n'
#define mem(a, b) memset(a, b, sizeof(a))

#define INF 0x3f3f3f3f

const LL mo = 19260817;

// char buf[1<<23],*p1=buf,*p2=buf,obuf[1<<23],*O=obuf; // 或者用fread更难调的快读
// #define getchar() (p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++)

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

template <class T>
void print(T x)
{
    if (x < 0)
    {
        x = -x;
        putchar('-');
        // *O++ = '-';
    }
    if (x > 9)
        print(x / 10);
    putchar(x % 10 + '0');
    // *O++ = x%10 + '0'
}
// fwrite(obuf,O-obuf,1,stdout);

template <typename IntegerType>
void convert_to_string(IntegerType x, std::string &s)
{
    if (x < 0)
    {
        x = -x;
        s.push_back('-');
        // *O++ = '-';
    }
    if (x > 9)
        convert_to_string(x / 10, s);
    s.push_back(x % 10 + '0');
}
namespace std
{
    string to_string(__int128 x)
    {
        string s;
        convert_to_string(x, s);
        return s;
    }
    istream &operator>>(istream &ins, __int128 &x)
    {
        x = 0;
        __int128 sgn = 1;
        int mono = ins.get();
        while (mono > '9' || mono < '0')
        {
            if (mono == '-')
                sgn = -sgn;
            mono = ins.get();
        }
        while (mono <= '9' && mono >= '0')
        {
            x = (x << 3) + (x << 1) + (mono ^ 0x30);
            mono = ins.get();
        }
        x *= sgn;
        ins.unget();
        return ins;
    }
    ostream &operator<<(ostream &ous, __int128 &x) { return ous << to_string(x); }
} // namespace std

// LL globalsgn = 1;
template <class T>
inline void qr(T &n)
{
    n = 0;
    int c = getchar();
    if (c == EOF)
    {
        throw "End of file!";
    }
    bool sgn = 0;
    // globalsgn = 1;

    while (c > '9' || c < '0')
    {
        if (c == '-')
            sgn ^= 1;
        c = getchar();
        if (c == EOF)
        {
            throw "End of file!";
        }
    }

    while (c <= '9' && c >= '0')
    {
        n = (n << 3) + (n << 1) + (c ^ 0x30);
        c = getchar();
    }
    // if (c == '-')
    //     globalsgn = -globalsgn;
    // n *= sgn;
    if (sgn)
        n = -n;
}

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
    return x >= 0 ? x : x + mo; // 为0时无解
}

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

inline LL CRT(LL factors[], LL remains[], LL length, LL prefix_mul)
{
    LL ans = 0;
    for (auto i = 0; i < length; i++)
    {
        LL tM = prefix_mul / factors[i];
        ans += remains[i] * (tM)*inv(tM, factors[i]);
        ans %= prefix_mul;
    }
    return ans;
}

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

// #define LUCASM 114514
#ifdef LUCASM
LL fact[LUCASM];

inline void get_fact(LL fact[], LL length) // 预处理阶乘用的
{
    fact[0] = 1;
    fact[1] = 1;
    for (auto i = 2; i < length; i++)
        fact[i] = fact[i - 1] * i % mo;
}

inline void get_fact(LL fact[], LL length, LL mo) // 预处理阶乘用的（求模
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

#endif

// #define FrontStar 666
#ifdef FrontStar

LL edge_cnt = 0;
struct EdgeModel
{
    LL next, to, val;
} eds[FrontStar];
LL node_ctr, merged_edge_ctr, edge_ctr;
LL head[FrontStar] = {-1};

inline void add(LL u, LL v)
{
    eds[edge_cnt].to = v;
    eds[edge_cnt].next = head[u];
    head[u] = edge_cnt++;
}

inline void add(LL u, LL v, LL w)
{
    eds[edge_cnt].to = v;
    eds[edge_cnt].val = w;
    eds[edge_cnt].next = head[u];
    head[u] = edge_cnt++;
}

#endif

// #define ORAFM 2333
#ifdef ORAFM
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

unordered_map<int, int> ans_mu;
unordered_map<int, LL> ans_phi;
// 杜教筛/数论分块
LL calc_phi_pref(LL x) // φ * I(常数函数) == id(恒等函数)
{
    if (x <= ORAFM)
        return phi[x];
    if (ans_phi.find(x) != ans_phi.end())
        return ans_phi[x];        // 缓存
    LL ret = (x + 1LL) * x / 2LL; // f*g即恒等函数的前缀和（等差数列求和公式
    for (unsigned int l = 2, r; l <= x; l = r + 1)
    {
        r = x / (x / l); // 跳到下取整位置
        ret -= (LL)(r - l + 1LL) * calc_phi_pref(x / l);
        // 依题意可能写成 ret -= (getsum(r) - getsum(l - 1)) * calc_mobius_pref(x / l); g的前缀和 即 getsum
    }
    return ans_phi[x] = ret;
}

int calc_mobius_pref(int x) // μ(mu, 莫比乌斯函数) * I(常数函数，恒等于1) == ε(单位函数，除n==1时为1外全为0)
{
    if (x <= ORAFM)
        return mobius[x];
    if (ans_mu.find(x) != ans_mu.end())
        return ans_mu[x];
    int ret = 1; // 单位函数的前缀就是1
    for (unsigned int l = 2, r; l <= x; l = r + 1)
    {
        r = x / (x / l);
        ret -= (r - l + 1) * calc_mobius_pref(x / l);
    }
    return ans_mu[x] = ret;
}
#endif

#ifdef KMPM
inline void Getnext(LL next[], char t[])
{
    LL p1 = 0;
    LL p2 = next[0] = -1;
    LL strlen_t = strlen(t);
    while (p1 < strlen_t)
    {
        if (p2 == -1 || t[p1] == t[p2])
            next[++p1] = ++p2;
        else
            p2 = next[p2];
    }
}

inline void KMP(char string[], char pattern[], LL next[])
{
    LL p1 = 0;
    LL p2 = 0;
    LL strlen_string = strlen(string);
    LL strlen_pattern = strlen(pattern);
    while (p1 < strlen_string)
    {
        if (p2 == -1 || string[p1] == pattern[p2])
            p1++, p2++;
        else
            p2 = next[p2];
        if (p2 == strlen_pattern)
            printf("%lld\n", p1 - strlen_pattern + 1), p2 = next[p2];
    }
}
#endif

// #define EXKMPM 25
#ifdef EXKMPM

string pattern;
string s;
LL nxt[EXKMPM];
LL extend[EXKMPM];

void getNEXT(string &pattern, LL next[])
{
    LL pLen = pattern.length();
    LL a = 0, k = 0;

    next[0] = pLen;
    for (auto i = 1; i < pLen; i++)
    {
        if (i >= k || i + next[i - a] >= k)
        {
            if (i >= k)
                k = i;
            while (k < pLen && pattern[k] == pattern[k - i])
                k++;
            next[i] = k - i;
            a = i;
        }
        else
        {
            next[i] = next[i - a];
        }
    }
}

void EXKMP(string &s, string &pattern, LL extend[], LL next[]) // string类得配O2不然过不了
{
    LL pLen = pattern.length();
    LL sLen = s.length();
    LL a = 0, k = 0;

    getNEXT(pattern, next);

    for (auto i = 0; i < sLen; i++)
    {
        if (i >= k || i + next[i - a] >= k)
        {
            if (i >= k)
                k = i;
            while (k < sLen && k - i < pLen && s[k] == pattern[k - i])
                k++;
            extend[i] = k - i;
            a = i;
        }
        else
        {
            extend[i] = next[i - a];
        }
    }
}

#endif

#define TreeArrayM 2000010

#ifdef TreeArrayM

#define lowbit(x) (x & -x)

struct TreeArray
{
    LL *content;
    LL len;
    TreeArray(LL len)
    {
        this->content = new LL[len];
        memset(this->content, 0, sizeof(LL) * len);
        this->len = len;
    }
    LL getsum(LL pos)
    {
        LL ans = 0;
        while (pos > 0)
        {
            ans += this->content[pos];
            pos -= lowbit(pos);
        }
        return ans;
    }
    void update(LL pos, LL x)
    {
        while (pos < this->len)
        {
            this->content[pos] += x;
            pos += lowbit(pos);
        }
    }
};
#endif

// #define Aho_CorasickAutomaton 2000010

#ifdef Aho_CorasickAutomaton

#define CharacterCount 26

struct TrieNode
{
    TrieNode *son[CharacterCount], *fail;
    // LL word_count;
    LL logs;

} T[Aho_CorasickAutomaton];
vector<TrieNode *> FailEdge[Aho_CorasickAutomaton];
LL AC_counter = 0;

vector<TrieNode *> trieIndex;

TrieNode *insertWords(string &s)
{
    auto root = &T[0];
    for (auto i : s)
    {
        auto nxt = i - 'a';
        if (root->son[nxt] == NULL)
            root->son[nxt] = &T[++AC_counter];
        root = root->son[nxt];
    }
    // word_count[root]++;

    return root; // 返回含单词的节点号
} // 用例：trieIndex.push_back(insertWords(s));

TrieNode *insertWords(char *s, LL &sLen)
{
    auto root = &T[0];
    for (auto i = 0; i < sLen; i++)
    {
        auto nxt = s[i] - 'a';
        if (root->son[nxt] == NULL)
            root->son[nxt] = &T[++AC_counter];
        root = root->son[nxt];
    }
    // word_count[root]++;

    return root; // 返回含单词的节点号
}

void getFail()
{
    queue<TrieNode *> Q; // bfs用
    for (auto i = 0; i < CharacterCount; i++)
    {
        if (T[0].son[i] != NULL)
        {
            T[0].son[i]->fail = &T[0];
            Q.push(T[0].son[i]);
        }
    }
    while (!Q.empty())
    {
        auto now = Q.front();
        Q.pop();
        now->fail = now->fail == NULL ? &T[0] : now->fail;
        for (auto i = 0; i < CharacterCount; i++)
        {
            if (now->son[i] != NULL)
            {
                now->son[i]->fail = now->fail->son[i];
                Q.push(now->son[i]);
            }
            else
            {
                now->son[i] = now->fail->son[i];
            }
        }
    }
} // 先设T[0].fail=0;所有单词插完以后调用一次

LL query(string &s)
{
    auto now = &T[0];
    auto ans = 0;
    for (auto i : s)
    {
        now = now->son[i - 'a'];
        now = now == NULL ? &T[0] : now;
        now->logs++;
        // for (auto j = now; j /*&& ~word_count[j]*/; j = fail[j])
        // {
        //     // ans += word_count[j];
        //     // cout << "j:" << j << endl;
        //     // if (word_count[j])
        //     logs[j]++;
        //     // for (auto k : word_position[j])
        //     //     pattern_count[k]++;
        //     // word_count[j] = -1; // 标记已经遍历的节点
        // }
    }

    for (auto i = 1; i <= AC_counter; i++)
    {
        FailEdge[T[i].fail - T].push_back(&T[i]);
    }

    return ans;
} // 查询母串，getFail后使用一次

LL query(char *s, LL &sLen)
{
    auto now = &T[0];
    auto ans = 0;
    for (auto i = 0; i < sLen; i++)
    {
        now = now->son[s[i] - 'a'];
        now = now == NULL ? &T[0] : now;
        now->logs++;
        // for (auto j = now; j /*&& ~word_count[j]*/; j = fail[j])
        // {
        // ans += word_count[j];
        // cout << "j:" << j << endl;
        // if (word_count[j])

        // for (auto k : word_position[j])
        //     pattern_count[k]++;
        // word_count[j] = -1; // 标记已经遍历的节点
        // }
    }

    for (auto i = 1; i <= AC_counter; i++)
    {
        FailEdge[T[i].fail - T].push_back(&T[i]);
    }

    return ans;
}

void AC_dfs(TrieNode *u)
{

    for (auto i : FailEdge[u - T])
    {
        AC_dfs(i);
        u->logs += i->logs;
    }
} // query完后使用，一般搜0号点

// 输出答案使用for(auto i:trieIndex)cout<<i.logs<<endl;这样

#endif

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

// #define XiongYaLi 1010
#ifdef XiongYaLi
LL pre_match[XiongYaLi];
LL searched[M];
bool dfs(LL x, const LL searchtime)
{
    if (searched[x] == searchtime)
        return false;
    searched[x] = searchtime;
    for (LL i = hds[x]; ~i; i = E[i].next)
    {
        LL obj = E[i].to;
        if (pre_match[obj] == -1 || dfs(pre_match[obj], searchtime))
        {
            pre_match[obj] = x;
            return true;
        }
    }
    return false;
}
LL get_max_match(LL lim)
{
    LL ans;
    for (LL i = 0 + 1; i < lim + 1; i++)
        ans += dfs(i, i);
    return ans;
}
#endif

// #define DINIC 520010

#ifdef DINIC
LL distant[M]; // 用于分层图层次表示
LL current[M]; // 当前弧优化
LL n, m, src, dst;

inline LL bfs()
{
    for (auto i = 1; i <= n; i++)
        distant[i] = INF;
    queue<LL> Q;
    Q.push(src);
    distant[src] = 0;
    current[src] = hds[src];
    while (!Q.empty())
    {
        auto x = Q.front();
        Q.pop();
        for (auto i = hds[x]; ~i; i = E[i].next)
        {
            auto v = E[i].to;
            if (E[i].val > 0 and distant[v] == INF)
            {
                Q.push(v);
                current[v] = hds[v];
                distant[v] = distant[x] + 1;
                if (v == dst)
                    return 1;
            }
        }
    }
    return 0;
}

inline LL dfs(LL x, LL Sum)
{
    if (x == dst)
        return Sum;
    LL res = 0;
    for (auto i = current[x]; ~i and Sum; i = E[i].next) // 当前弧优化：改变枚举起点
    {
        current[x] = i;
        auto v = E[i].to;
        if (E[i].val > 0 and (distant[v] == distant[x] + 1))
        {
            LL remain = dfs(v, min(Sum, E[i].val)); // remain:当前最小剩余的流量
            if (remain == 0)
                distant[v] = INF; // 去掉增广完毕的点
            E[i].val -= remain;
            E[i ^ 1].val += remain;
            res += remain; // 经过该点的所有流量和
            Sum -= remain; // 该点剩余流量
        }
    }
    return res;
}

LL Dinic()
{
    LL ans = 0;
    while (bfs())
        ans += dfs(src, INF);
    return ans;
}
#endif

// https://www.cnblogs.com/bosswnx/p/10570783.html
template <typename T>
struct MCMF // 费用流(Dinic)zkw板子
{           // Based on Dinic (zkw)

    typedef long long LL;
    T INF;
    int N = 1e5 + 5; // 最大点meta参数，要按需改
#define _N 100006
    std::bitset<_N> vis; // 要一起改
    T *Dis;
    int s, t;           // 源点，汇点需要外部写入
    int *Cur;           // 当前弧优化用
    T maxflow, mincost; // 放最终答案

    struct EdgeContent
    {
        int to;
        T flow;
        T cost;
        int dualEdge;
        EdgeContent(int a, T b, T c, int d) : to(a), flow(b), cost(c), dualEdge(d) {}
    };

    std::vector<EdgeContent> *E;

    MCMF(int n)
    {
        N = n;
        E = new std::vector<EdgeContent>[n + 1];
        Dis = new T[n + 1];
        Cur = new int[n + 1];
        maxflow = mincost = 0;
        memset(&INF, 0x3f, sizeof(INF));
    }

    void add(int u, int v, T f, T w) // 加一条u到v流为f单位费为w的边
    {
        E[u].emplace_back(v, f, w, E[v].size());
        E[v].emplace_back(u, 0, -w, E[u].size() - 1);
    }

    bool SPFA()
    {
        std::queue<int> Q;
        Q.emplace(s);
        memset(Dis, INF, sizeof(T) * (N + 1));
        Dis[s] = 0;
        int k;
        while (!Q.empty())
        {
            k = Q.front();
            Q.pop();
            vis.reset(k);
            for (auto [to, f, w, rev] : E[k])
            {
                if (f and Dis[k] + w < Dis[to])
                {
                    Dis[to] = Dis[k] + w;
                    if (!vis.test(to))
                    {
                        Q.emplace(to);
                        vis.set(to);
                    }
                }
            }
        }
        return Dis[t] != INF;
    }
    T DFS(int k, T flow)
    {
        if (k == t)
        {
            maxflow += flow;
            return flow;
        }
        T sum = 0;
        vis.set(k);
        for (auto i = Cur[k]; i < E[k].size(); i++)
        {
            auto &[to, f, w, rev] = E[k][i];
            if (!vis.test(to) and f and Dis[to] == Dis[k] + w)
            {
                Cur[k] = i;
                T p = DFS(to, std::min(flow - sum, f));
                sum += p;
                f -= p;
                E[to][rev].flow += p;
                mincost += p * w;
                if (sum == flow)
                    break;
            }
        }
        vis.reset(k);
        return sum;
    }

    void Dinic() // 入口
    {
        while (SPFA())
        {
            memset(Cur, 0, sizeof(int) * (N + 1));
            DFS(s, INF);
        }
    }
}; // namespace MCMF

struct range
{
    struct _iterator
    {
        LL _start;
        LL _step;
        _iterator(LL from, LL step) : _start(from), _step(step) {}
        _iterator(LL from) : _start(from), _step(1) {}
        inline bool sign(LL x) { return x < 0; }
        bool operator!=(_iterator &b) { return _start != b._start and sign(b._start - _start) == sign(_step); }
        LL operator*() { return _start; }
        _iterator &operator++()
        {
            _start += _step;
            return *this;
        }
    };
    _iterator _finish;
    _iterator _begin;
    range(LL to) : _begin(0), _finish(to) {}
    range(LL from, LL to) : _begin(from), _finish(to) {}
    range(LL from, LL to, LL step) : _begin(from, step), _finish(to, step) {}
    _iterator &begin() { return _begin; }
    _iterator &end() { return _finish; }
};

struct subset
{
    struct _iterator
    {
        LL _start;
        LL _father;
        _iterator(LL from, LL step) : _start(from), _father(step) {}
        bool operator!=(_iterator &b) { return _start != b._start; }
        LL operator*() { return _start; }
        _iterator &operator++()
        {
            _start = (_start - 1) & _father;
            return *this;
        }
    };
    _iterator _finish;
    _iterator _begin;
    subset(LL father) : _begin(father, father), _finish(0, father) {}
    _iterator &begin() { return _begin; }
    _iterator &end() { return _finish; }
};

/* 用例
#include <iostream>
signed main()
{
    int n, m, s, t;
    std::cin >> n >> m >> s >> t;
    auto MMP = MCMF<LL>(n);
    MMP.s = s;
    MMP.t = t;
    for (auto i : range(m))
    {
        int u, v;
        LL flow, cost;
        std::cin >> u >> v >> flow >> cost;
        MMP.add(u, v, flow, cost);
    }
    MMP.Dinic();
    std::cout << MMP.maxflow << ' ' << MMP.mincost << std::endl;

    return 0;
}*/

#ifdef HLPP_ENABLE
/* 除非卡时不然别用的预流推进桶排序优化黑魔法，用例如下
signed main()
{
	qr(HLPP::n);
	// qr(HLPP::m);
	qr(HLPP::src);
	qr(HLPP::dst);
	while (HLPP::m--)
	{
		LL t1, t2, t3;
		qr(t1);
		qr(t2);
		qr(t3);
		HLPP::add(t1, t2, t3);
	}
	cout << HLPP::hlpp(HLPP::n + 1, HLPP::src, HLPP::dst) << endl;
	return 0;
}
*/
namespace HLPP
{
    const LL INF = 0x3f3f3f3f3f3f;
    const LL MXn = 1203;
    const LL maxm = 520010;

    vector<LL> gap;
    LL n, src, dst, now_height, src_height;

    struct NODEINFO
    {
        LL height = MXn, traffic;
        LL getIndex();
        NODEINFO(LL h = 0) : height(h) {}
        bool operator<(const NODEINFO &a) const { return height < a.height; }
    } node[MXn];

    LL NODEINFO::getIndex() { return this - node; }

    struct EDGEINFO
    {
        LL to;
        LL flow;
        LL opposite;
        EDGEINFO(LL a, LL b, LL c) : to(a), flow(b), opposite(c) {}
    };
    std::list<NODEINFO *> dlist[MXn];
    vector<std::list<NODEINFO *>::iterator> iter;
    vector<NODEINFO *> list[MXn];
    vector<EDGEINFO> edge[MXn];

    inline void add(LL u, LL v, LL w = 0)
    {
        edge[u].push_back(EDGEINFO(v, w, (LL)edge[v].size()));
        edge[v].push_back(EDGEINFO(u, 0, (LL)edge[u].size() - 1));
    }

    priority_queue<NODEINFO> PQ;
    inline bool prework_bfs(NODEINFO &src, NODEINFO &dst, LL &n)
    {
        gap.assign(n, 0);
        for (auto i = 0; i <= n; i++)
            node[i].height = n;
        dst.height = 0;
        queue<NODEINFO *> q;
        q.push(&dst);
        while (!q.empty())
        {
            NODEINFO &top = *(q.front());
            for (auto i : edge[&top - node])
            {
                if (node[i.to].height == n and edge[i.to][i.opposite].flow > 0)
                {
                    gap[node[i.to].height = top.height + 1]++;
                    q.push(&node[i.to]);
                }
            }
            q.pop();
        }

        return src.height == n;
    }

    inline void relabel(NODEINFO &src, NODEINFO &dst, LL &n)
    {
        prework_bfs(src, dst, n);
        for (auto i = 0; i <= n; i++)
            list[i].clear(), dlist[i].clear();

        for (auto i = 0; i <= n; i++)
        {
            NODEINFO &u = node[i];
            if (u.height < n)
            {
                iter[i] = dlist[u.height].insert(dlist[u.height].begin(), &u);
                if (u.traffic > 0)
                    list[u.height].push_back(&u);
            }
        }
        now_height = src_height = src.height;
    }

    inline bool push(NODEINFO &u, EDGEINFO &dst) // 从x到y尽可能推流，p是边的编号
    {
        NODEINFO &v = node[dst.to];
        LL w = min(u.traffic, dst.flow);
        dst.flow -= w;
        edge[dst.to][dst.opposite].flow += w;
        u.traffic -= w;
        v.traffic += w;
        if (v.traffic > 0 and v.traffic <= w)
            list[v.height].push_back(&v);
        return u.traffic;
    }

    inline void push(LL n, LL ui)
    {
        auto new_height = n;
        NODEINFO &u = node[ui];
        for (auto &i : edge[ui])
        {
            if (i.flow)
            {
                if (u.height == node[i.to].height + 1)
                {
                    if (!push(u, i))
                        return;
                }
                else
                    new_height = min(new_height, node[i.to].height + 1); // 抬到正好流入下一个点
            }
        }
        auto height = u.height;
        if (gap[height] == 1)
        {
            for (auto i = height; i <= src_height; i++)
            {
                for (auto it : dlist[i])
                {
                    gap[(*it).height]--;
                    (*it).height = n;
                }
                dlist[i].clear();
            }
            src_height = height - 1;
        }
        else
        {
            gap[height]--;
            iter[ui] = dlist[height].erase(iter[ui]);
            u.height = new_height;
            if (new_height == n)
                return;
            gap[new_height]++;
            iter[ui] = dlist[new_height].insert(dlist[new_height].begin(), &u);
            src_height = max(src_height, now_height = new_height);
            list[new_height].push_back(&u);
        }
    }

    inline LL hlpp(LL n, LL s, LL t)
    {
        if (s == t)
            return 0;
        now_height = src_height = 0;
        NODEINFO &src = node[s];
        NODEINFO &dst = node[t];
        iter.resize(n);
        for (auto i = 0; i < n; i++)
            if (i != s)
                iter[i] = dlist[node[i].height].insert(dlist[node[i].height].begin(), &node[i]);
        gap.assign(n, 0);
        gap[0] = n - 1;
        src.traffic = INF;
        dst.traffic = -INF; // 上负是为了防止来自汇点的推流
        for (auto &i : edge[s])
            push(src, i);
        src.traffic = 0;
        relabel(src, dst, n);
        for (LL ui; now_height >= 0;)
        {
            if (list[now_height].empty())
            {
                now_height--;
                continue;
            }
            NODEINFO &u = *(list[now_height].back());
            list[now_height].pop_back();
            push(n, &u - node);
        }
        return dst.traffic + INF;
    }
} // namespace HLPP

#endif
int clz(int N)
{
    return N ? 32 - __builtin_clz(N) : -INF;
}
int clz(unsigned long long N)
{
    return N ? 64 - __builtin_clzll(N) : -INF;
}

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

// min_25结束

template <typename TYP>
struct PersistentSengmentTree
{
    static const int size_elapsed = 9000000; // 因为需要静态分配，在这里指定预估最大大小

    static inline int mid(int lower, int upper) { return (lower + upper) >> 1; };

    int cursiz = 0;
    int l_bound_cache, r_bound_cache;
    struct Node
    {
        TYP meta;
        int l, r;
        Node() : meta(), l(-1), r(-1) {}
        void at_build()
        {
            // memset(meta, 0, sizeof(meta));
            meta = 0;
            l = -1;
            r = -1;
        }
    } nodes[size_elapsed];

    int headers[size_elapsed];
    int h_pointer = 0;

    void init()
    {
        h_pointer = 0;
        cursiz = 0;
    }

    // std::vector<Node> nodes; // 动态开点的选项
    // PersistentSengmentTree() : cursiz(0) { nodes.resize(size_elapsed); }

    inline int _build(int l_bound, int r_bound)
    {
        int cur_num = cursiz++;
        Node &me = nodes[cur_num];
        me.at_build();
        if (r_bound > l_bound)
        {
            int m = mid(l_bound, r_bound);
            me.l = _build(l_bound, m);
            me.r = _build(m + 1, r_bound);
        }
        return cur_num;
    }

    void build(int _n)
    {
        headers[h_pointer++] = _build(1, _n);
        l_bound_cache = 1 - 1;
        r_bound_cache = _n - 1;
    }

    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &&updval)
    {
        int cur_num = cursiz++;
        nodes[cur_num] = nodes[before];
        Node &me = nodes[cur_num];
        // 这里改更新策略
        me.meta += updval;
        //
        // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
        if (l_bound < r_bound)
        {
            int m = mid(l_bound, r_bound);
            if (pos <= m) // 值域线段树，落在哪边就往哪边走
                me.l = _update(l_bound, m, pos, me.l, updval);
            else
                me.r = _update(m + 1, r_bound, pos, me.r, updval);
        }
        return cur_num;
    }
    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &updval) { return _update(l_bound, r_bound, pos, before, std::move(updval)); }

    void update(int pos, TYP &&updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }
    void update(int pos, TYP &updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }

    /*查询的rank是排名，返回的是离散化的排好序的序列的下标，查询函数根据业务需求改*/
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &&rank)
    {
        Node &u = nodes[front_node];
        Node &v = nodes[back_node];
        if (l_bound >= r_bound)
            return l_bound;
        TYP lx = nodes[v.l].meta - nodes[u.l].meta;
        // TYP rx = nodes[v.r].meta - nodes[u.r].meta;
        // TYP lx = nodes[v.l].meta;
        // cerr << "vlmeta:" << nodes[v.l].meta << "\tulmeta:" << nodes[u.l].meta << endl;
        // cerr << "vrmeta:" << nodes[v.r].meta << "\turmeta:" << nodes[u.r].meta << endl;
        int m = mid(l_bound, r_bound);
        if (lx >= rank)
            return _query(l_bound, m, u.l, v.l, rank);
        else //if (2 * rx > rank)
            return _query(m + 1, r_bound, u.r, v.r, rank - lx);
        // return 0;
    }
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &rank) { return _query(l_bound, r_bound, front_node, back_node, std::move(rank)); }

    int query(int l, int r, TYP &&k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }
    int query(int l, int r, TYP &k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }

    int legacy_query(int l_bound, int r_bound, int ql, int qr, int cur)
    {
        Node &me = nodes[cur];
        if (l_bound >= ql and r_bound <= qr)
            return me.meta;
        int res = 0;
        int m = mid(l_bound, r_bound);
        if (ql <= m)
            res += legacy_query(l_bound, m, ql, qr, me.l);
        if (qr >= m + 1)
            res += legacy_query(m + 1, r_bound, ql, qr, me.r);
        return res;
    }
};

// #define MODULO MO

#define REQUIRE_RMQ

namespace Tree
{
    template <typename T>
    struct _iNode
    {
        T lazy_add;
        T sum_content;
        T lazy_mul;
#ifdef REQUIRE_RMQ
        T max_content;
        T min_content;
#endif
    };

    // template <typename T>
    // struct type_deduce
    // {
    //     using typ = T;
    // };

    template <typename T> // 使用的类T必须支持= （问就是你常用的int, long long, double甚至__int128都是支持的）
    struct SegmentTree
    {
        using _Node = _iNode<T>;
        int len;        // 线段树实际节点数
        int valid_len;  // 原有效数据长度
        _Node *_start;  // 起始位置
        _Node *_finish; // 结束位置
        // template <typename AllocationPlaceType = void>
        SegmentTree(int length, void *arr = nullptr) // 构造函数只分配内存
        {
            valid_len = length;
            len = 1 << 1 + (int)ceil(log2(length));

            // while (length > 1)
            // {
            //     len += length;
            //     length = length + 1 >> 1;
            // }

            if (arr != nullptr)
            {
                _start = ::new (arr) _Node[len]; // 会占用arr序列的空间
            }
            else
            {
                _start = new _Node[len];
            }

            _finish = _start + len;
        }

        _Node *begin() { return _start; }
        _Node *end() { return _finish; }

        static int mid(int l, int r) { return l + r >> 1; }

        void show()
        {
            std::cout << '[';
            for (_Node *i = begin(); i != end(); i++)
                std::cout << i->sum_content << ",]"[i == end() - 1] << " \n"[i == end() - 1];
        }

        std::function<void(int, T &&, int)> update_policies[2] =
            {
                [&](int x, T &&v, int my_length)
                {
                    _start[x].lazy_add *= v; // 更新此次修改的tag值
                    _start[x].sum_content *= v;
                    _start[x].lazy_mul *= v;
#ifdef REQUIRE_RMQ
                    _start[x].max_content *= v;
                    _start[x].min_content *= v;
#endif
#ifdef MODULO
                    _start[x].lazy_mul %= MODULO;
                    _start[x].sum_content %= MODULO;
                    _start[x].lazy_add %= MODULO;
#endif
                },
                [&](int x, T &&v, int my_length)
                {
                    _start[x].lazy_add += v; // 更新此次修改的tag值
                    _start[x].sum_content += my_length * v;
#ifdef REQUIRE_RMQ
                    _start[x].max_content += v;
                    _start[x].min_content += v;
#endif
#ifdef MODULO
                    _start[x].sum_content %= MODULO;
                    _start[x].lazy_add %= MODULO;
#endif
                }};

        std::function<void(int, T &)> query_policies[3] =
            {
                [&](int x, T &res)
                {
                    res += _start[x].sum_content;
#ifdef MODULO
                    res %= MODULO;
#endif
                },
                [&](int x, T &res)
                {
#ifdef REQUIRE_RMQ
                    res = min(res, _start[x].min_content);
#endif
                },
                [&](int x, T &res)
                {
#ifdef REQUIRE_RMQ
                    res = max(res, _start[x].max_content);
#endif
                }};

        template <typename Func>
        void range_update(
            int l,
            int r,
            T &&v,
            int node_l,
            int node_r,
            int x,
            Func &update_policy)
        {
            if (l <= node_l and node_r <= r)
            {
                update_policy(x, std::move(v), node_r - node_l + 1);
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    range_update(l, r, std::move(v), node_l, mi, x << 1, update_policy);
                if (r > mi)
                    range_update(l, r, std::move(v), mi + 1, node_r, x << 1 | 1, update_policy);
                maintain(x);
            }
        }

        void range_mul(int l, int r, T &v)
        {
            range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[0]);
        }

        void range_mul(int l, int r, T &&v)
        {
            range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[0]);
        }

        void range_add(int l, int r, T &v)
        {
            range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[1]);
        }

        void range_add(int l, int r, T &&v)
        {
            range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[1]);
        }

        inline void maintain(int i)
        {
            int l = i << 1;
            int r = l | 1;
            _start[i].sum_content = (_start[l].sum_content + _start[r].sum_content)
#ifdef MODULO
                                    % MODULO
#endif
                ;
#ifdef REQUIRE_RMQ
            _start[i].max_content = max(_start[l].max_content, _start[r].max_content);
            _start[i].min_content = min(_start[l].min_content, _start[r].min_content);
#endif
        }

        void assign(T values[]) { build(values, 1, valid_len, 1); }

        inline void build(T values[], int l, int r, int x)
        {
            _start[x].lazy_add = 0;
            _start[x].lazy_mul = 1;
            if (l == r)
            {
                _start[x].sum_content = values[l - 1];
#ifdef REQUIRE_RMQ
                _start[x].max_content = values[l - 1];
                _start[x].min_content = values[l - 1];
#endif
            }
            else
            {
                int mi = mid(l, r);
                build(values, l, mi, x << 1);
                build(values, mi + 1, r, x << 1 | 1);
                maintain(x);
            }
        }

        inline void push_down(int ind, int my_left_bound, int my_right_bound)
        {
            int l = ind << 1;
            int r = l | 1;
            int mi = mid(my_left_bound, my_right_bound);
            int lson_length = (mi - my_left_bound + 1);
            int rson_length = (my_right_bound - mi);
            if (_start[ind].lazy_mul != 1)
            {
                _start[l].sum_content *= _start[ind].lazy_mul;
                _start[l].sum_content += _start[ind].lazy_add * lson_length;

                _start[r].sum_content *= _start[ind].lazy_mul;
                _start[r].sum_content += _start[ind].lazy_add * rson_length;

                _start[l].lazy_mul *= _start[ind].lazy_mul;
                _start[l].lazy_add *= _start[ind].lazy_mul;
                _start[l].lazy_add += _start[ind].lazy_add;

                _start[r].lazy_mul *= _start[ind].lazy_mul;
                _start[r].lazy_add *= _start[ind].lazy_mul;
                _start[r].lazy_add += _start[ind].lazy_add;
#ifdef MODULO
                _start[l].lazy_mul %= MODULO;
                _start[l].lazy_add %= MODULO;
                _start[l].sum_content %= MODULO;

                _start[r].lazy_mul %= MODULO;
                _start[r].lazy_add %= MODULO;
                _start[r].sum_content %= MODULO;
#endif

#ifdef REQUIRE_RMQ
                _start[l].max_content *= _start[ind].lazy_mul;
                _start[l].max_content += _start[ind].lazy_add;
                _start[l].min_content *= _start[ind].lazy_mul;
                _start[l].min_content += _start[ind].lazy_add;

                _start[r].max_content *= _start[ind].lazy_mul;
                _start[r].max_content += _start[ind].lazy_add;
                _start[r].min_content *= _start[ind].lazy_mul;
                _start[r].min_content += _start[ind].lazy_add;
#endif
                _start[ind].lazy_mul = 1;
                _start[ind].lazy_add = 0;

                return;
            }
            if (_start[ind].lazy_add)
            {
                _start[l].sum_content += _start[ind].lazy_add * lson_length;
                _start[l].lazy_add += _start[ind].lazy_add;
                _start[r].sum_content += _start[ind].lazy_add * rson_length;
                _start[r].lazy_add += _start[ind].lazy_add;
#ifdef MODULO
                _start[l].lazy_add %= MODULO;
                _start[l].sum_content %= MODULO;
                _start[r].lazy_add %= MODULO;
                _start[r].sum_content %= MODULO;
#endif

#ifdef REQUIRE_RMQ
                _start[l].max_content += _start[ind].lazy_add;
                _start[l].min_content += _start[ind].lazy_add;

                _start[r].max_content += _start[ind].lazy_add;
                _start[r].min_content += _start[ind].lazy_add;
#endif
                _start[ind].lazy_add = 0;
            }
        }

        template <typename Func>
        void query_proxy(
            int l,
            int r,
            T &res,
            int node_l,
            int node_r,
            int x,
            Func &query_policy)
        {
            if (l <= node_l and node_r <= r)
            {
                query_policy(x, res);
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    query_proxy(l, r, res, node_l, mi, x << 1, query_policy);
                if (r > mi)
                    query_proxy(l, r, res, mi + 1, node_r, x << 1 | 1, query_policy);
                maintain(x);
            }
        }

        T query_sum(int l, int r)
        {
            T res = 0;
            query_proxy(l, r, res, 1, valid_len, 1, query_policies[0]);
            return res;
        }
#ifdef REQUIRE_RMQ
        T query_max(int l, int r)
        {
            T res = 0;
            query_proxy(l, r, res, 1, valid_len, 1, query_policies[2]);
            return res;
        }

        T query_min(int l, int r)
        {
            T res;
            memset(&res, 0x3f, sizeof(res));
            query_proxy(l, r, res, 1, valid_len, 1, query_policies[1]);
            return res;
        }
#endif
    };

} // namespace Tree

#define ComputeGeometry
#ifdef ComputeGeometry
namespace Geometry
{
    using FLOAT_ = double;

    constexpr const FLOAT_ Infinity = INFINITY;
    const FLOAT_ decimal_round = 1e-8; // 精度参数

    const FLOAT_ DEC = 1.0 / decimal_round;
    const double smooth_const2 = 0.479999989271164; // 二次项平滑系数
    const double smooth_const3 = 0.234999999403954; // 三次项平滑系数

    const FLOAT_ PI = acos(-1);
    bool round_compare(FLOAT_ a, FLOAT_ b) { return round(DEC * a) == round(DEC * b); }
    FLOAT_ Round(FLOAT_ a) { return round(DEC * a) / DEC; }

    template <typename PrecisionType = LL>
    struct Fraction
    {
        PrecisionType upper, lower;

        Fraction(PrecisionType u = 0, PrecisionType l = 1)
        {
            upper = u;
            lower = l;
        }
        void normalize()
        {
            if (upper)
            {
                PrecisionType g = abs(std::__gcd(upper, lower));
                upper /= g;
                lower /= g;
            }
            else
                lower = 1;
            if (lower < 0)
            {
                lower = -lower;
                upper = -upper;
            }
        }
        long double ToFloat() { return (long double)upper / (long double)lower; }
        bool operator==(Fraction b) { return upper * b.lower == lower * b.upper; }
        bool operator>(Fraction b) { return upper * b.lower > lower * b.upper; }
        bool operator<(Fraction b) { return upper * b.lower < lower * b.upper; }
        bool operator<=(Fraction b) { return !(*this > b); }
        bool operator>=(Fraction b) { return !(*this < b); }
        bool operator!=(Fraction b) { return !(*this == b); }
        Fraction operator-() { return Fraction(-upper, lower); }
        Fraction operator+(Fraction b) { return Fraction(upper * b.lower + b.upper * lower, lower * b.lower); }
        Fraction operator-(Fraction b) { return (*this) + (-b); }
        Fraction operator*(Fraction b) { return Fraction(upper * b.upper, lower * b.lower); }
        Fraction operator/(Fraction b) { return Fraction(upper * b.lower, lower * b.upper); }
        Fraction &operator+=(Fraction b)
        {
            *this = *this + b;
            this->normalize();
            return *this;
        }
        Fraction &operator-=(Fraction b)
        {
            *this = *this - b;
            this->normalize();
            return *this;
        }
        Fraction &operator*=(Fraction b)
        {
            *this = *this * b;
            this->normalize();
            return *this;
        }
        Fraction &operator/=(Fraction b)
        {
            *this = *this / b;
            this->normalize();
            return *this;
        }
        friend Fraction fabs(Fraction a) { return Fraction(abs(a.upper), abs(a.lower)); }
        std::string to_string() { return lower == 1 ? std::to_string(upper) : std::to_string(upper) + '/' + std::to_string(lower); }
        friend std::ostream &operator<<(std::ostream &o, Fraction a)
        {
            return o << "Fraction(" << std::to_string(a.upper) << ", " << std::to_string(a.lower) << ")";
        }
        friend std::istream &operator>>(std::istream &i, Fraction &a)
        {
            char slash;
            return i >> a.upper >> slash >> a.lower;
        }
        friend isfinite(Fraction a) { return a.lower != 0; }
        void set_value(PrecisionType u, PrecisionType d = 1) { upper = u, lower = d; }
    };

    template <typename VALUETYPE = FLOAT_>
    struct VectorN : std::vector<VALUETYPE>
    {
        void init(int siz, VALUETYPE default_val = 0)
        {
            this->clear();
            this->assign(siz, default_val);
            this->resize(siz);
        }
        VectorN(int siz, VALUETYPE default_val = 0) { init(siz, default_val); }
        VectorN() = default;

        bool any()
        {
            for (auto &i : *this)
            {
                if (i != 0)
                    return true;
                else if (!isfinite(i))
                    return false;
            }
            return false;
        }
        bool is_all_nan()
        {
            for (auto &i : *this)
            {
                if (!isnan(i))
                    return false;
            }
            return true;
        }

        void rconcat(VectorN &&r)
        {
            this->insert(this->end(), r.begin(), r.end());
        }
        void rconcat(VectorN &r) { rconcat(std::move(r)); }

        void lerase(int ctr)
        {
            assert(this->size() >= ctr);
            this->erase(this->begin(), this->begin() + ctr);
        }

        // 四则运算赋值重载
        VectorN &operator*=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i *= operand;
            return (*this);
        }
        VectorN &operator*=(VALUETYPE &operand) { return (*this) *= std::move(operand); }

        VectorN &operator/=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i /= operand;
            return (*this);
        }
        VectorN &operator/=(VALUETYPE &operand) { return (*this) /= std::move(operand); }

        VectorN &operator%=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i = (i % operand + operand) % operand;
            return (*this);
        }
        VectorN &operator%=(VALUETYPE &operand) { return (*this) %= std::move(operand); }

        VectorN &operator-=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i -= operand;
            return (*this);
        }
        VectorN &operator-=(VALUETYPE &operand) { return (*this) -= std::move(operand); }

        VectorN &operator+=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i += operand;
            return (*this);
        }
        VectorN &operator+=(VALUETYPE &operand) { return (*this) += std::move(operand); }
        // 结束

        // 四则运算 和单个数

        VectorN operator*(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * operand;
            return ret;
        }
        VectorN operator*(VALUETYPE &operand) { return (*this) * std::move(operand); }
        friend VectorN operator*(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand * r[i];
            return ret;
        }
        friend VectorN operator*(VALUETYPE &operand, VectorN &r) { return std::move(operand) * r; }

        VectorN operator/(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / operand;
            return ret;
        }
        VectorN operator/(VALUETYPE &operand) { return (*this) / std::move(operand); }
        friend VectorN operator/(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand / r[i];
            return ret;
        }
        friend VectorN operator/(VALUETYPE &operand, VectorN &r) { return std::move(operand) / r; }

        VectorN operator+(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + operand;
            return ret;
        }
        VectorN operator+(VALUETYPE &operand) { return (*this) + std::move(operand); }
        friend VectorN operator+(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand + r[i];
            return ret;
        }
        friend VectorN operator+(VALUETYPE &operand, VectorN &r) { return std::move(operand) + r; }

        VectorN operator-(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - operand;
            return ret;
        }
        VectorN operator-(VALUETYPE &operand) { return (*this) - std::move(operand); }
        friend VectorN operator-(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand - r[i];
            return ret;
        }
        friend VectorN operator-(VALUETYPE &operand, VectorN &r) { return std::move(operand) - r; }

        /*不推荐使用的转发算子*/
        template <typename ANYRHS>
        VectorN operator+(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator-(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator*(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator/(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / rhs;
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator+(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs + rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator-(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs - rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator*(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs * rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator/(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs / rhs[i];
            return ret;
        }

        // 结束

        // 四则运算 和同类

        VectorN operator+(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + operand[i];
            return ret;
        }
        VectorN operator+(VectorN &operand) { return *this + std::move(operand); }

        VectorN operator-(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - operand[i];
            return ret;
        }
        VectorN operator-(VectorN &operand) { return *this - std::move(operand); }

        VectorN operator*(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * operand[i];
            return ret;
        }
        VectorN operator*(VectorN &operand) { return *this * std::move(operand); }

        VectorN operator/(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / operand[i];
            return ret;
        }
        VectorN operator/(VectorN &operand) { return *this / std::move(operand); }

        // 结束

        // 赋值算子

        VectorN &operator+=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] += operand[i];
            return (*this);
        }
        VectorN &operator-=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] -= operand[i];
            return (*this);
        }
        VectorN &operator*=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] *= operand[i];
            return (*this);
        }
        VectorN &operator/=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] /= operand[i];
            return (*this);
        }

        VectorN &operator+=(VectorN &operand) { return (*this) += std::move(operand); }
        VectorN &operator-=(VectorN &operand) { return (*this) -= std::move(operand); }
        VectorN &operator*=(VectorN &operand) { return (*this) *= std::move(operand); }
        VectorN &operator/=(VectorN &operand) { return (*this) /= std::move(operand); }

        std::string ToString()
        {
            std::ostringstream ostr;
            for (int i = 0; i < this->size(); i++)
            {
                ostr << std::setw(8) << std::scientific << std::right;
                ostr << (*this)[i] << (i == this->size() - 1 ? "\n" : ", ");
            }
            return ostr.str();
        }
        friend std::ostream &operator<<(std::ostream &o, VectorN &m) { return o << m.ToString(); }
        friend std::ostream &operator<<(std::ostream &o, VectorN &&m) { return o << m.ToString(); }

        /* 向量的平方模 */
        VALUETYPE sqrMagnitude()
        {
            VALUETYPE res = 0;
            for (auto &i : (*this))
                res += i * i;
            return res;
        }

        /* 向量的模 */
        VALUETYPE magnitude() { return sqrt(this->sqrMagnitude()); }

        /* 向量单位化 */
        void Normalize()
        {
            VALUETYPE _m = this->magnitude();
            for (auto &i : (*this))
                i /= _m;
        }

        /* 返回与该向量方向同向的单位向量 */
        VectorN normalized()
        {
            VectorN ret(*this);
            ret.Normalize();
            return ret;
        }

        /* 距离 */
        static VALUETYPE Distance(VectorN &a, VectorN &b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &a, VectorN &&b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &&a, VectorN &b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &&a, VectorN &&b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static VectorN LerpUnclamped(VectorN &a, VectorN &b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &a, VectorN &&b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &&a, VectorN &b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &&a, VectorN &&b, VALUETYPE t) { return a + (b - a) * t; }

        /* 点积 */
        static VALUETYPE Dot(VectorN &lhs, VectorN &rhs) { return Dot(std::move(lhs), std::move(rhs)); }
        static VALUETYPE Dot(VectorN &lhs, VectorN &&rhs) { return Dot(std::move(lhs), rhs); }
        static VALUETYPE Dot(VectorN &&lhs, VectorN &rhs) { return Dot(lhs, std::move(rhs)); }
        static VALUETYPE Dot(VectorN &&lhs, VectorN &&rhs)
        {
            VALUETYPE ans = 0;
            for (auto i = 0; i < lhs._len; i++)
                ans += lhs.data[i] * rhs.data[i];
            return ans;
        }

        /*无符号弧度夹角*/
        static VALUETYPE Rad(VectorN &from, VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &from, VectorN &&to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &&from, VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &&from, VectorN &&to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /*无符号角度夹角*/
        static VALUETYPE Angle(VectorN &from, VectorN &to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &from, VectorN &&to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &&from, VectorN &to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &&from, VectorN &&to) { return Rad(from, to) * 180.0 / PI; }

        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static VectorN Max(VectorN lhs, VectorN &&rhs)
        {
            for (auto &&i : range(lhs._len))
                lhs.data[i] = std::max(lhs.data[i], rhs.data[i]);
            return lhs;
        }
        static VectorN Max(VectorN lhs, VectorN &rhs) { return Max(lhs, std::move(rhs)); }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static VectorN Min(VectorN lhs, VectorN &&rhs)
        {
            for (auto &&i : range(lhs._len))
                lhs.data[i] = std::min(lhs.data[i], rhs.data[i]);
            return lhs;
        }
        static VectorN Min(VectorN lhs, VectorN &rhs) { return Min(lhs, std::move(rhs)); }

        /*获得vector在onNormal方向的投影*/
        static VectorN Project(VectorN &vector, VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &vector, VectorN &&onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &&vector, VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &&vector, VectorN &&onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
    };

    template <typename VALUETYPE = FLOAT_>
    struct Matrix : VectorN<VectorN<VALUETYPE>>
    {
        int ROW, COL;
        // std::vector<VectorN<VALUETYPE>> data;

        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Matrix" << ROW << "x" << COL << "[\n";
            for (auto &i : *this)
                ostr << '\t' << i.ToString();
            ostr << "]";
            return ostr.str();
        }

        friend std::ostream &operator<<(std::ostream &o, Matrix &m) { return o << m.ToString(); }
        friend std::ostream &operator<<(std::ostream &o, Matrix &&m) { return o << m.ToString(); }

        Matrix(VectorN<VectorN<VALUETYPE>> &&v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}
        Matrix(VectorN<VectorN<VALUETYPE>> &v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}

        Matrix(int r, int c, VALUETYPE default_val = 0) : ROW(r), COL(c)
        {
            this->resize(r);
            for (r--; r >= 0; r--)
                (*this)[r].init(c, default_val);
        }
        Matrix() = default;

        /*交换两行*/
        void swap_rows(int from, int to)
        {
            std::swap((*this)[from], (*this)[to]);
        }

        /*化为上三角矩阵*/
        void triangularify(bool unitriangularify = false)
        {
            int mx;
            int done_rows = 0;
            for (int j = 0; j < COL; j++) // 化为上三角
            {
                mx = done_rows;
                for (int i = done_rows + 1; i < ROW; i++)
                {
                    if (fabs((*this)[i][j]) > fabs((*this)[mx][j]))
                        mx = i;
                }
                if ((*this)[mx][j] == 0)
                    continue;
                if (mx != done_rows)
                    swap_rows(mx, done_rows);

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    VALUETYPE tmp = (*this)[i][j] / (*this)[done_rows][j];
                    if (tmp != 0)
                        (*this)[i] -= (*this)[done_rows] * tmp;
                }
                if (unitriangularify)
                {
                    auto tmp = (*this)[done_rows][j];
                    (*this)[done_rows] /= tmp; // 因为用了引用，这里得拷贝暂存
                }
                done_rows++;
                if (done_rows == ROW)
                    break;
            }
        }

        /*化为上三角矩阵，模意义版*/
        void triangularify(long long mod, bool unitriangularify = false)
        {
            int mx;
            int done_rows = 0;
            for (int j = 0; j < COL; j++) // 化为上三角
            {
                mx = done_rows;

                if ((*this)[done_rows][j] < 0)
                    (*this)[done_rows][j] = ((*this)[done_rows][j] % mod + mod) % mod;

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    if ((*this)[i][j] < 0)
                        (*this)[i][j] = ((*this)[i][j] % mod + mod) % mod;
                    if ((*this)[i][j] > (*this)[mx][j])
                        mx = i;
                }
                if ((*this)[mx][j] == 0)
                    continue;

                if (mx != done_rows)
                    swap_rows(mx, done_rows);

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    VALUETYPE tmp = (*this)[i][j] * inv((*this)[done_rows][j], mod) % mod;
                    if (tmp != 0)
                    {
                        (*this)[i] -= (*this)[done_rows] * tmp;
                        (*this)[i] %= mod;
                    }
                }
                if (unitriangularify)
                {
                    auto tmp = (*this)[done_rows][j];
                    (*this)[done_rows] *= inv(tmp, mod);
                    (*this)[done_rows] %= mod;
                }
                done_rows++;
                if (done_rows == ROW)
                    break;
            }
        }

        void row_echelonify(long long mod = 0)
        {
            if (mod)
                triangularify(mod, true);
            else
                triangularify(true);
            int valid_pos = 1;
            for (int i = 1; i < ROW; i++)
            {
                while (valid_pos < COL and (*this)[i][valid_pos] == 0)
                    valid_pos++;
                if (valid_pos == COL)
                    break;
                for (int ii = i - 1; ii >= 0; ii--)
                {
                    (*this)[ii] -= (*this)[i] * (*this)[ii][valid_pos];
                    if (mod)
                        (*this)[ii] %= mod;
                }
            }
        }

        /*返回一个自身化为上三角矩阵的拷贝*/
        Matrix triangular(bool unitriangularify = false)
        {
            Matrix ret(*this);
            ret.triangularify(unitriangularify);
            return ret;
        }

        /*求秩，得先上三角化*/
        int _rank()
        {
            int res = 0;
            for (auto &i : (*this))
                res += i.any();
            return res;
        }

        /*求秩*/
        int rank() { return triangular()._rank(); }

        /*高斯消元解方程组*/
        bool solve()
        {
            if (COL != ROW + 1)
                throw "dimension error!";
            triangularify();
            // cerr << *this << endl;
            if (!(*this).back().any())
                return false;
            for (int i = ROW - 1; i >= 0; i--)
            {
                for (int j = i + 1; j < ROW; j++)
                    (*this)[i][COL - 1] -= (*this)[i][j] * (*this)[j][COL - 1];
                if ((*this)[i][i] == 0)
                    return false;
                (*this)[i][COL - 1] /= (*this)[i][i];
            }
            return true;
        }

        /*矩阵连接*/
        void rconcat(Matrix &&rhs)
        {
            COL += rhs.COL;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].rconcat(rhs[i]);
            }
        }
        void rconcat(Matrix &rhs) { rconcat(std::move(rhs)); }

        /*左截断*/
        void lerase(int ctr)
        {
            assert(COL >= ctr);
            COL -= ctr;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].lerase(ctr);
            }
        }

        /*矩阵乘法*/
        Matrix dot(Matrix &&rhs, long long mod = 0)
        {
            if (this->COL != rhs.ROW)
                throw "Error at matrix multiply: lhs's column is not equal to rhs's row";
            Matrix ret(this->ROW, rhs.COL, true);
            for (int i = 0; i < ret.ROW; ++i)
                for (int k = 0; k < this->COL; ++k)
                {
                    VALUETYPE &s = (*this)[i][k];
                    for (int j = 0; j < ret.COL; ++j)
                    {
                        ret[i][j] += s * rhs[k][j];
                        if (mod)
                            ret[i][j] %= mod;
                    }
                }
            return ret;
        }
        Matrix dot(Matrix &rhs, long long mod = 0) { return dot(std::move(rhs), mod); }
    };

    template <typename VALUETYPE = FLOAT_>
    struct SquareMatrix : Matrix<VALUETYPE>
    {
        SquareMatrix(int siz, bool is_reset = false) : Matrix<VALUETYPE>(siz, siz, is_reset) {}
        SquareMatrix(Matrix<VALUETYPE> &&x) : Matrix<VALUETYPE>(x)
        {
            assert(x.COL == x.ROW);
        }
        static SquareMatrix eye(int siz)
        {
            SquareMatrix ret(siz, true);
            for (siz--; siz >= 0; siz--)
                ret[siz][siz] = 1;
            return ret;
        }

        SquareMatrix quick_power(long long p, long long mod = 0)
        {
            SquareMatrix ans = eye(this->ROW);
            SquareMatrix rhs(*this);
            while (p)
            {
                if (p & 1)
                {
                    ans = ans.dot(rhs, mod);
                }
                rhs = rhs.dot(rhs, mod);
                p >>= 1;
            }
            return ans;
        }

        SquareMatrix inv(long long mod = 0)
        {
            Matrix<VALUETYPE> ret(*this);
            ret.rconcat(eye(this->ROW));
            ret.row_echelonify(mod);
            // cerr << ret << endl;
            for (int i = 0; i < this->ROW; i++)
            {
                if (ret[i][i] != 1)
                    throw "Error at matrix inverse: cannot identify extended matrix";
            }
            ret.lerase(this->ROW);
            return ret;
        }
    };

    struct Vector2
    {
        FLOAT_ x, y;
        Vector2(FLOAT_ _x, FLOAT_ _y) : x(_x), y(_y) {}
        Vector2(FLOAT_ n) : x(n), y(n) {}
        Vector2() : x(0.0), y(0.0) {}
        Vector2 &operator=(Vector2 b)
        {
            this->x = b.x;
            this->y = b.y;
            return *this;
        }
        bool operator==(Vector2 b) { return round_compare(this->x, b.x) and round_compare(this->y, b.y); }
        bool operator!=(Vector2 b) { return not((*this) == b); }
        FLOAT_ &operator[](int ind)
        {
            switch (ind)
            {
            case 0:
                return this->x;
                break;
            case 1:
                return this->y;
                break;
            case 'x':
                return this->x;
                break;
            case 'y':
                return this->y;
                break;
            default:
                throw "无法理解除0,1外的索引";
                break;
            }
        }
        friend std::ostream &operator<<(std::ostream &o, Vector2 v)
        {
            o << v.ToString();
            return o;
        }
        friend Vector2 operator*(FLOAT_ n, Vector2 v) { return Vector2(v.x * n, v.y * n); }
        friend Vector2 operator/(FLOAT_ n, Vector2 v) { return Vector2(n / v.x, n / v.y); }
        Vector2 operator-() { return Vector2(-(this->x), -(this->y)); }
        Vector2 operator+(Vector2 b) { return Vector2(this->x + b.x, this->y + b.y); }
        Vector2 operator-(Vector2 b) { return (*this) + (-b); }
        Vector2 operator*(FLOAT_ n) { return Vector2(this->x * n, this->y * n); }
        Vector2 operator*(Vector2 b) { return Vector2(this->x * b.x, this->y * b.y); }
        Vector2 operator/(FLOAT_ n) { return (*this) * (FLOAT_(1) / n); }
        Vector2 operator/(Vector2 b) { return (*this) * (FLOAT_(1) / b); }
        Vector2 operator+=(Vector2 b) { return (*this) = (*this) + b; }

        /* 向量的平方模 */
        FLOAT_ sqrMagnitude() { return pow(this->x, 2) + pow(this->y, 2); }
        /* 向量的模 */
        FLOAT_ magnitude() { return sqrt(this->sqrMagnitude()); }
        /*判等*/
        bool equals(Vector2 b) { return (*this) == b; }

        /*用极坐标换算笛卡尔坐标*/
        static Vector2 fromPolarCoordinate(Vector2 v, bool use_degree = 1)
        {
            return v.toCartesianCoordinate(use_degree);
        }

        /*转为笛卡尔坐标*/
        Vector2 toCartesianCoordinate(bool use_degree = 1)
        {
            return Vector2(
                x * cos(y * (use_degree ? PI / 180.0 : 1)),
                x * sin(y * (use_degree ? PI / 180.0 : 1)));
        }
        /*转为极坐标*/
        Vector2 toPolarCoordinate(bool use_degree = 1)
        {
            return Vector2(
                magnitude(),
                atan2(y, x) * (use_degree ? 180.0 / PI : 1));
        }

        /*获取极角*/
        FLOAT_ toPolarAngle(bool use_degree = 1)
        {
            return atan2(y, x) * (use_degree ? 180.0 / PI : 1);
        }

        /*转为极坐标*/
        static Vector2 ToPolarCoordinate(Vector2 coordinate, bool use_degree = 1) { return coordinate.toPolarCoordinate(use_degree); }

        static bool Equals(Vector2 a, Vector2 b) { return a == b; }
        /* 向量单位化 */
        void Normalize()
        {
            FLOAT_ _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
        }
        /*设置值*/
        void Set(FLOAT_ newX, FLOAT_ newY)
        {
            this->x = newX;
            this->y = newY;
        }
        /*转为字符串*/
        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Vector2(" << this->x << ", " << this->y << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量 */
        Vector2 normalized()
        {
            FLOAT_ _m = this->magnitude();
            return Vector2(this->x / _m, this->y / _m);
        }
        // FLOAT_ Distance(Vector2 b) { return ((*this) - b).magnitude(); }
        /* 距离 */
        static FLOAT_ Distance(Vector2 a, Vector2 b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static Vector2 LerpUnclamped(Vector2 a, Vector2 b, FLOAT_ t)
        {
            Vector2 c = b - a;
            return a + c * t;
        }

        /* 向量线性插值,t限制于[0,1] */
        static Vector2 Lerp(Vector2 a, Vector2 b, FLOAT_ t)
        {
            t = max(FLOAT_(0), t);
            t = min(FLOAT_(1), t);
            return Vector2::LerpUnclamped(a, b, t);
        }

        /* 向量线性插值,t限制于(-无穷,1] */
        static Vector2 MoveTowards(Vector2 current, Vector2 target, FLOAT_ maxDistanceDelta)
        {
            maxDistanceDelta = min(FLOAT_(1), maxDistanceDelta);
            return Vector2::LerpUnclamped(current, target, maxDistanceDelta);
        }

        /* 拿它的垂直向量（逆时针旋转90°） */
        static Vector2 Perpendicular(Vector2 inDirection)
        {
            return Vector2(-inDirection.y, inDirection.x);
        }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        static Vector2 Reflect(Vector2 inDirection, Vector2 inNormal)
        {
            return inDirection - 2 * Vector2::Dot(inDirection, inNormal) * inNormal;
        }

        /* 点积 */
        static FLOAT_ Dot(Vector2 lhs, Vector2 rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y;
        }
        /* 叉积 */
        static FLOAT_ Cross(Vector2 lhs, Vector2 rhs) { return lhs.x * rhs.y - lhs.y * rhs.x; }

        /* 对位相乘罢了 */
        static Vector2 Scale(Vector2 a, Vector2 b) { return Vector2(a.x * b.x, a.y * b.y); }

        /* 对位相乘罢了 */
        Vector2 Scale(Vector2 scale) { return (*this) * scale; }

        /*有符号弧度夹角*/
        static FLOAT_ SignedRad(Vector2 from, Vector2 to) { return atan2(Vector2::Cross(from, to), Vector2::Dot(from, to)); }
        /*无符号弧度夹角*/
        static FLOAT_ Rad(Vector2 from, Vector2 to) { return abs(Vector2::SignedRad(from, to)); }
        /*有符号角度夹角*/
        static FLOAT_ SignedAngle(Vector2 from, Vector2 to)
        {
            // return acos(Vector2::Dot(from, to) / (from.magnitude() * to.magnitude()));
            return Vector2::SignedRad(from, to) * 180.0 / PI;
        }
        /*无符号角度夹角*/
        static FLOAT_ Angle(Vector2 from, Vector2 to) { return abs(Vector2::SignedAngle(from, to)); }

        /*返回该方向上最大不超过maxLength长度的向量*/
        static Vector2 ClampMagnitude(Vector2 vector, FLOAT_ maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static Vector2 Max(Vector2 lhs, Vector2 rhs)
        {
            return Vector2(max(lhs.x, rhs.x), max(lhs.y, rhs.y));
        }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static Vector2 Min(Vector2 lhs, Vector2 rhs)
        {
            return Vector2(min(lhs.x, rhs.x), min(lhs.y, rhs.y));
        }

        /*获得vector在onNormal方向的投影*/
        static Vector2 Project(Vector2 vector, Vector2 onNormal)
        {
            return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal;
        }

    };

    struct Vector3 // 三维向量
    {
        FLOAT_ x, y, z;
        Vector3(FLOAT_ _x, FLOAT_ _y, FLOAT_ _z) : x(_x), y(_y), z(_z) {}
        Vector3(FLOAT_ n) : x(n), y(n), z(n) {}
        Vector3(Vector2 &b) : x(b.x), y(b.y), z(0.0) {}
        // Vector3(Vector2 &&b) : x(b.x), y(b.y), z(0.0) {}
        Vector3() : x(0.0), y(0.0), z(0.0) {}
        Vector3 &operator=(Vector3 b)
        {
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            return *this;
        }
        bool operator==(Vector3 b) { return round_compare(this->x, b.x) and round_compare(this->y, b.y) and round_compare(this->z, b.z); }
        bool operator!=(Vector3 b) { return not((*this) == b); }
        FLOAT_ &operator[](int ind)
        {
            switch (ind)
            {
            case 0:
                return this->x;
                break;
            case 1:
                return this->y;
                break;
            case 2:
                return this->z;
                break;
            case 'x':
                return this->x;
                break;
            case 'y':
                return this->y;
                break;
            case 'z':
                return this->z;
            default:
                throw "无法理解除0,1,2外的索引";
                break;
            }
        }
        friend std::ostream &operator<<(std::ostream &o, Vector3 v)
        {
            o << v.ToString();
            return o;
        }
        friend Vector3 operator*(FLOAT_ n, Vector3 v) { return Vector3(v.x * n, v.y * n, v.z * n); }
        friend Vector3 operator/(FLOAT_ n, Vector3 v) { return Vector3(n / v.x, n / v.y, n / v.z); }
        Vector3 operator-() { return Vector3(-(this->x), -(this->y), -(this->z)); }
        Vector3 operator+(Vector3 b) { return Vector3(this->x + b.x, this->y + b.y, this->z + b.z); }
        Vector3 operator-(Vector3 b) { return (*this) + (-b); }
        Vector3 operator*(FLOAT_ n) { return Vector3(this->x * n, this->y * n, this->z * n); }
        Vector3 operator*(Vector3 b) { return Vector3(this->x * b.x, this->y * b.y, this->z * b.z); }
        Vector3 operator/(FLOAT_ n) { return (*this) * (FLOAT_(1) / n); }
        Vector3 operator/(Vector3 b) { return (*this) * (FLOAT_(1) / b); }
        Vector3 operator-=(Vector3 b) { return (*this) = (*this) - b; }

        /* 向量的平方模 */
        FLOAT_ sqrMagnitude() { return pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2); }
        /* 向量的模 */
        FLOAT_ magnitude() { return sqrt(this->sqrMagnitude()); }
        /*判等*/
        bool equals(Vector3 b) { return (*this) == b; }
        static bool Equals(Vector3 a, Vector3 b) { return a == b; }
        /* 向量单位化 */
        void Normalize()
        {
            FLOAT_ _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
            this->z /= _m;
        }
        /*设置值*/
        void Set(FLOAT_ newX, FLOAT_ newY, FLOAT_ newZ)
        {
            this->x = newX;
            this->y = newY;
            this->z = newZ;
        }
        /*转为字符串*/
        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Vector3(" << this->x << ", " << this->y << ", " << this->z << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量 */
        Vector3 normalized()
        {
            FLOAT_ _m = this->magnitude();
            return Vector3(this->x / _m, this->y / _m, this->z / _m);
        }
        // FLOAT_ Distance(Vector2 b) { return ((*this) - b).magnitude(); }
        /* 距离 */
        static FLOAT_ Distance(Vector3 a, Vector3 b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static Vector3 LerpUnclamped(Vector3 a, Vector3 b, FLOAT_ t) { return a + (b - a) * t; }

        /* 向量线性插值,t限制于[0,1] */
        static Vector3 Lerp(Vector3 a, Vector3 b, FLOAT_ t)
        {
            t = max(FLOAT_(0), t);
            t = min(FLOAT_(1), t);
            return Vector3::LerpUnclamped(a, b, t);
        }

        /* 向量线性插值,t限制于(-无穷,1] */
        static Vector3 MoveTowards(Vector3 current, Vector3 target, FLOAT_ maxDistanceDelta)
        {
            maxDistanceDelta = min(FLOAT_(1), maxDistanceDelta);
            return Vector3::LerpUnclamped(current, target, maxDistanceDelta);
        }

        /* 拿它的垂直向量（逆时针旋转90°） */
        static Vector3 Perpendicular(Vector3 inDirection)
        {
            return Vector3(-inDirection.y, inDirection.x, 0);
        }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        static Vector3 Reflect(Vector3 inDirection, Vector3 inNormal)
        {
            return inDirection - 2 * Vector3::Dot(inDirection, inNormal) * inNormal;
        }

        /* 点积 */
        static FLOAT_ Dot(Vector3 lhs, Vector3 rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }
        /* 叉积 */
        static Vector3 Cross(Vector3 lhs, Vector3 rhs) { return Vector3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }

        /* 对位相乘罢了 */
        static Vector3 Scale(Vector3 a, Vector3 b) { return a * b; }

        /* 对位相乘罢了 */
        Vector3 Scale(Vector3 scale) { return (*this) * scale; }

        /*无符号弧度夹角*/
        static FLOAT_ Rad(Vector3 from, Vector3 to) { return acos(Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /*无符号角度夹角*/
        static FLOAT_ Angle(Vector3 from, Vector3 to) { return Rad(from, to) * 180 / PI; }

        /*返回该方向上最大不超过maxLength长度的向量*/
        static Vector3 ClampMagnitude(Vector3 vector, FLOAT_ maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static Vector3 Max(Vector3 lhs, Vector3 rhs)
        {
            return Vector3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z));
        }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static Vector3 Min(Vector3 lhs, Vector3 rhs)
        {
            return Vector3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z));
        }

        /*获得vector在onNormal(请自己单位化)方向的投影*/
        static Vector3 Project(Vector3 vector, Vector3 onNormal)
        {
            return vector.magnitude() * cos(Rad(vector, onNormal)) * onNormal;
        }

        /*将两个向量单位化，并调整切线位置使之垂直于法向*/
        static void OrthoNormalize(Vector3 &normal, Vector3 &tangent)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
        }

        /*将三个向量单位化，并调整使之两两垂直*/
        static void OrthoNormalize(Vector3 &normal, Vector3 &tangent, Vector3 &binormal)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
            binormal -= Project(binormal, normal);
            binormal -= Project(binormal, tangent);
            binormal.Normalize();
        }

        /*获得vector在以planeNormal为法向量的平面的投影*/
        static Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal)
        {
            FLOAT_ mag = vector.magnitude();
            FLOAT_ s = Rad(vector, planeNormal);
            OrthoNormalize(planeNormal, vector);
            return mag * sin(s) * vector;
        }

        /*罗德里格旋转公式，获得current绕轴normal(请自己单位化)旋转degree度（默认角度）的向量，右手螺旋意义*/
        static Vector3 Rotate(Vector3 current, Vector3 normal, FLOAT_ degree, bool use_degree = 1)
        {
            FLOAT_ r = use_degree ? degree / 180 * PI : degree;
            FLOAT_ c = cos(r);
            return c * current + (1.0 - c) * Dot(normal, current) * normal + Cross(sin(r) * normal, current);
        }

        /*将current向target转向degree度，如果大于夹角则返回target方向长度为current的向量*/
        static Vector3 RotateTo(Vector3 current, Vector3 target, FLOAT_ degree, bool use_degree = 1)
        {
            FLOAT_ r = use_degree ? degree / 180 * PI : degree;
            if (r >= Rad(current, target))
                return current.magnitude() / target.magnitude() * target;
            else
            {
                // FLOAT_ mag = current.magnitude();
                Vector3 nm = Cross(current, target).normalized();
                return Rotate(current, nm, r);
            }
        }

        /*球面插值*/
        static Vector3 SlerpUnclamped(Vector3 a, Vector3 b, FLOAT_ t)
        {
            Vector3 rot = RotateTo(a, b, Rad(a, b) * t, false);
            FLOAT_ l = b.magnitude() * t + a.magnitude() * (1 - t);
            return rot.normalized() * l;
        }

        /*球面插值，t限制于[0,1]版本*/
        static Vector3 Slerp(Vector3 a, Vector3 b, FLOAT_ t)
        {
            if (t <= 0)
                return a;
            if (t >= 1)
                return b;
            return SlerpUnclamped(a, b, t);
        }
        /*根据经纬，拿一个单位化的三维向量，以北纬和东经为正*/
        static Vector3 FromLongitudeAndLatitude(FLOAT_ longitude, FLOAT_ latitude)
        {
            Vector3 lat = Rotate(Vector3(1, 0, 0), Vector3(0, -1, 0), latitude);
            return Rotate(lat, Vector3(0, 0, 1), longitude);
        }

        /*球坐标转换为xyz型三维向量*/
        static Vector3 FromSphericalCoordinate(Vector3 spherical, bool use_degree = 1) { return FromSphericalCoordinate(spherical.x, spherical.y, spherical.z, use_degree); }
        /*球坐标转换为xyz型三维向量，半径r，theta倾斜角（纬度），phi方位角（经度），默认输出角度*/
        static Vector3 FromSphericalCoordinate(FLOAT_ r, FLOAT_ theta, FLOAT_ phi, bool use_degree = 1)
        {
            theta = use_degree ? theta / 180 * PI : theta;
            phi = use_degree ? phi / 180 * PI : phi;
            return Vector3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        static Vector3 ToSphericalCoordinate(Vector3 coordinate, bool use_degree = 1)
        {
            FLOAT_ r = coordinate.magnitude();
            return Vector3(
                r,
                acos(coordinate.z / r) * (use_degree ? 180.0 / PI : 1),
                atan2(coordinate.y, coordinate.x) * (use_degree ? 180.0 / PI : 1));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        Vector3 toSphericalCoordinate(bool use_degree = 1) { return ToSphericalCoordinate(*this, use_degree); }

    };

    struct Line2
    {
        FLOAT_ A, B, C;
        Line2(Vector2 u, Vector2 v) : A(u.y - v.y), B(v.x - u.x), C(u.y * (u.x - v.x) - u.x * (u.y - v.y))
        {
            if (u == v)
            {
                if (u.x)
                {
                    A = 1;
                    B = 0;
                    C = -u.x;
                }
                else if (u.y)
                {
                    A = 0;
                    B = 1;
                    C = -u.y;
                }
                else
                {
                    A = 1;
                    B = -1;
                    C = 0;
                }
            }
        }
        Line2(FLOAT_ a, FLOAT_ b, FLOAT_ c) : A(a), B(b), C(c) {}
        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Line2(" << this->A << ", " << this->B << ", " << this->C << ")";
            return ostr.str();
        }
        friend std::ostream &operator<<(std::ostream &o, Line2 v)
        {
            o << v.ToString();
            return o;
        }
        FLOAT_ k() { return -A / B; }
        FLOAT_ b() { return -C / B; }
        FLOAT_ x(FLOAT_ y) { return -(B * y + C) / A; }
        FLOAT_ y(FLOAT_ x) { return -(A * x + C) / B; }
        /*点到直线的距离*/
        FLOAT_ distToPoint(Vector2 p) { return abs(A * p.x + B * p.y + C / sqrt(A * A + B * B)); }
        /*直线距离公式，使用前先判平行*/
        static FLOAT_ Distance(Line2 a, Line2 b) { return abs(a.C - b.C) / sqrt(a.A * a.A + a.B * a.B); }
        /*判断平行*/
        static bool IsParallel(Line2 u, Line2 v)
        {
            bool f1 = round_compare(u.B, 0.0);
            bool f2 = round_compare(v.B, 0.0);
            if (f1 != f2)
                return false;
            return f1 or round_compare(u.k(), v.k());
        }

        /*单位化（？）*/
        void normalize()
        {
            FLOAT_ su = sqrt(A * A + B * B + C * C);
            if (A < 0)
                su = -su;
            else if (A == 0 and B < 0)
                su = -su;
            A /= su;
            B /= su;
            C /= su;
        }
        /*返回单位化后的直线*/
        Line2 normalized()
        {
            Line2 t(*this);
            t.normalize();
            return t;
        }

        bool operator==(Line2 v) { return round_compare(A, v.A) and round_compare(B, v.B) and round_compare(C, v.C); }
        bool operator!=(Line2 v) { return !(*this == v); }

        static bool IsSame(Line2 u, Line2 v)
        {
            return Line2::IsParallel(u, v) and round_compare(Distance(u.normalized(), v.normalized()), 0.0);
        }

        /*计算交点*/
        static Vector2 Intersect(Line2 u, Line2 v)
        {
            FLOAT_ tx = (u.B * v.C - v.B * u.C) / (v.B * u.A - u.B * v.A);
            FLOAT_ ty = (u.B != 0.0 ? -u.A * tx / u.B - u.C / u.B : -v.A * tx / v.B - v.C / v.B);
            return Vector2(tx, ty);
        }

        /*判断三个点是否共线*/
        static bool Collinear(Vector2 a, Vector2 b, Vector2 c)
        {
            Line2 l(a, b);
            return round_compare((l.A * c.x + l.B * c.y + l.C), 0.0);
        }
    };

    struct Segment2 : Line2 // 二维有向线段
    {
        Vector2 from, to;
        Segment2(Vector2 a, Vector2 b) : Line2(a, b), from(a), to(b) {}
        Segment2(FLOAT_ x, FLOAT_ y, FLOAT_ X, FLOAT_ Y) : Line2(Vector2(x, y), Vector2(X, Y)), from(Vector2(x, y)), to(Vector2(X, Y)) {}
        bool is_online(Vector2 poi)
        {
            return round_compare((Vector2::Distance(poi, to) + Vector2::Distance(poi, from)), Vector2::Distance(from, to));
        }
        Vector2 &operator[](int i)
        {
            switch (i)
            {
            case 0:
                return from;
                break;
            case 1:
                return to;
                break;
            default:
                throw "数组越界";
                break;
            }
        }
    };

    struct Polygon
    {
        std::vector<Vector2> points;

    private:
        Vector2 accordance;
        // bool anticlock_comparator(Vector2 &a, Vector2 &b)
        // {
        //     return Vector2::SignedRad(accordance, a) < Vector2::SignedRad(accordance, b);
        // }

    public:
        /*凸多边形用逆时针排序*/
        void autoanticlockwiselize()
        {
            accordance = average();
            anticlockwiselize();
        }

        // typedef bool(Polygon::*comparator);

        void anticlockwiselize()
        {
            // comparator cmp = &Polygon::anticlock_comparator;
            auto anticlock_comparator = [&](Vector2 &a, Vector2 &b) -> bool
            {
                return (a - accordance).toPolarCoordinate(false).y < (b - accordance).toPolarCoordinate(false).y;
            };
            sort(points.begin(), points.end(), anticlock_comparator);
            // for (auto i : points)
            // {
            //     cerr << (i - accordance).toPolarCoordinate() << "\t" << i << endl;
            // }
        }

        Vector2 average()
        {
            Vector2 avg(0, 0);
            for (auto i : points)
            {
                avg += i;
            }
            return avg / points.size();
        }

        /*求周长*/
        FLOAT_ perimeter()
        {
            FLOAT_ ret = Vector2::Distance(points.front(), points.back());
            for (int i = 1; i < points.size(); i++)
                ret += Vector2::Distance(points[i], points[i - 1]);
            return ret;
        }
        /*面积*/
        FLOAT_ area()
        {
            FLOAT_ ret = Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + Vector2::Cross(points[i - 1], points[i]);
            return ret / 2;
        }
        /*求几何中心（形心、重心）*/
        Vector2 center()
        {
            Vector2 ret = (points.back() + points.front()) * Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + (points[i - 1] + points[i]) * Vector2::Cross(points[i - 1], points[i]);
            return ret / area() / 6;
        }
        /*求边界整点数*/
        long long boundary_points()
        {
            long long b = 0;
            for (int i = 0; i < points.size() - 1; i++)
            {
                b += std::__gcd((long long)abs(points[i + 1].x - points[i].x), (long long)abs(points[i + 1].y - points[i].y));
            }
            return b;
        }
        /*Pick定理：多边形面积=内部整点数+边界上的整点数/2-1；求内部整点数*/
        long long interior_points(FLOAT_ A = -1, long long b = -1)
        {
            if (A < 0)
                A = area();
            if (b < 0)
                b = boundary_points();
            return (long long)A + 1 - (b / 2);
        }
    };

} // namespace Geometry
#endif
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/assoc_container.hpp>
__gnu_pbds::tree<int, __gnu_pbds::null_type, std::less<int>, __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update> TTT;

// 函数不返回值可能会 RE
// 少码大数据结构，想想复杂度更优的做法
// 小数 二分/三分 注意break条件
// 浮点运算 sqrt(a^2-b^2) 可用 sqrt(a+b)*sqrt(a-b) 代替，避免精度问题
// long double -> %Lf 别用C11 (C14/16)
// 控制位数 cout << setprecision(10) << ans;
// reverse vector 注意判空 不然会re
// 分块注意维护块上标记 来更新块内数组a[]
// vector+lower_bound常数 < map/set/(unordered_map)
// map.find不会创建新元素 map[]会 注意空间
// 别对指针用memset
// 用位运算表示2^n注意加LL 1LL<<20
// 注意递归爆栈
// 注意边界
// 注意memset 多组会T

// lambda

// sort(p + 1, p + 1 + n,
//              [](const point &x, const point &y) -> bool { return x.x < y.x; });

// append l1 to l2 (l1 unchanged)

// l2.insert(l2.end(),l1.begin(),l1.end());

// append l1 to l2 (elements appended to l2 are removed from l1)
// (general form ... TG gave form that is actually better suited
//  for your needs)

// l2.splice(l2.end(),l1,l1.begin(),l1.end());