// cpp -dD -P -fpreprocessed | tr -d  '[:space:]' | md5sum

#define __AVX__ 1
#define __AVX2__ 1
#define __SSE__ 1
#define __SSE2__ 1
#define __SSE2_MATH__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSSE3__ 1
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")
#include <immintrin.h>
#include <emmintrin.h>
// -Wl,--stack=256000000
// #pragma GCC optimize(1)
// #pragma GCC optimize(2)
// #pragma GCC optimize(3)
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
// #pragma GCC optimize("Ofast")
// #pragma GCC optimize("inline")
// #pragma GCC optimize("-fgcse")
// #pragma GCC optimize("-fgcse-lm")
// #pragma GCC optimize("-fipa-sra")
// #pragma GCC optimize("-ftree-pre")
// #pragma GCC optimize("-ftree-vrp")
// #pragma GCC optimize("-fpeephole2")
// #pragma GCC optimize("-ffast-math")
// #pragma GCC optimize("-fsched-spec")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC optimize("-falign-jumps")
// #pragma GCC optimize("-falign-loops")
// #pragma GCC optimize("-falign-labels")
// #pragma GCC optimize("-fdevirtualize")
// #pragma GCC optimize("-fcaller-saves")
// #pragma GCC optimize("-fcrossjumping")
// #pragma GCC optimize("-fthread-jumps")
// #pragma GCC optimize("-funroll-loops")
// #pragma GCC optimize("-fwhole-program")
// #pragma GCC optimize("-freorder-blocks")
// #pragma GCC optimize("-fschedule-insns")
// #pragma GCC optimize("inline-functions")
// #pragma GCC optimize("-ftree-tail-merge")
// #pragma GCC optimize("-fschedule-insns2")
// #pragma GCC optimize("-fstrict-aliasing")
// #pragma GCC optimize("-fstrict-overflow")
// #pragma GCC optimize("-falign-functions")
// #pragma GCC optimize("-fcse-skip-blocks")
// #pragma GCC optimize("-fcse-follow-jumps")
// #pragma GCC optimize("-fsched-interblock")
// #pragma GCC optimize("-fpartial-inlining")
// #pragma GCC optimize("no-stack-protector")
// #pragma GCC optimize("-freorder-functions")
// #pragma GCC optimize("-findirect-inlining")
// #pragma GCC optimize("-fhoist-adjacent-loads")
// #pragma GCC optimize("-frerun-cse-after-loop")
// #pragma GCC optimize("inline-small-functions")
// #pragma GCC optimize("-finline-small-functions")
// #pragma GCC optimize("-ftree-switch-conversion")
// #pragma GCC optimize("-foptimize-sibling-calls")
// #pragma GCC optimize("-fexpensive-optimizations")
// #pragma GCC optimize("-funsafe-loop-optimizations")
// #pragma GCC optimize("inline-functions-called-once")
// #pragma GCC optimize("-fdelete-null-pointer-checks")
// __builtin_popcount(); // 数1
// __builtin_clz(); // 数前导零
// __builtin_ctz(); // 数后导零

// #define in ,
// #define foreach(...) foreach_ex(foreach_in, (__VA_ARGS__))
// #define foreach_ex(m, wrapped_args) m wrapped_args
// #define foreach_in(e, a) for (int i = 0, elem *e = a->elems; i != a->size; i++, e++)

#define sign(_x) (_x < 0)
#define range_4(__iter__, __from__, __to__, __step__) for (LL __iter__ = __from__; __iter__ != __to__ && sign(__to__ - __from__) == sign(__step__); __iter__ += __step__)
#define range_3(__iter__, __from__, __to__) range_4(__iter__, __from__, __to__, 1)
#define range_2(__iter__, __to__) range_4(__iter__, 0, __to__, 1)
#define range_1(__iter__, __to__) range_4(__iter__, 0, 1, 1)
#define get_range(_1, _2, _3, _4, _Func, ...) _Func
#define crange(...) get_range(__VA_ARGS__, range_4, range_3, range_2, range_1, ...)(__VA_ARGS__)

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
}

int clz(int N)
{
    return N ? 32 - __builtin_clz(N) : -INF;
}
int clz(unsigned long long N)
{
    return N ? 64 - __builtin_clzll(N) : -INF;
}

int log2int(int x) { return 31 - __builtin_clz(x); }
int log2ll(long long x) { return 63 - __builtin_clzll(x); }

// 扩栈


extern int main2(void) __asm__ ("_main2");
 
int main2() {
    char test[255 << 20];
    memset(test, 42, sizeof(test));
    printf(":)\n");
    exit(0); // 得使用exit0退出，不能使用return 0
}
 
int main() {
    int size = 256 << 20;  // 256Mb
    char *p = (char *)malloc(size) + size;
    __asm__ __volatile__(
        "movl  %0, %%esp\n"
        "pushl $_exit\n" 
        "jmp _main2\n"
        :: "r"(p));
    // __asm__ __volatile__(
    //     "movq  %0, %%rsp\n"
    //     "pushq $exit\n" 
    //     "jmp main2\n"
    //     :: "r"(p));
}

template<typename FT>
FT randreal(FT begin, FT end)
{
    static std::default_random_engine eng(time(0));
    std::uniform_real_distribution<FT> skip_rate(begin, end);
    return skip_rate(eng);
}

template<typename IT>
IT randint(IT begin, IT end)
{
    static std::default_random_engine eng(time(0));
    std::uniform_int_distribution<IT> skip_rate(begin, end);
    return skip_rate(eng);
}

std::chrono::_V2::steady_clock::time_point C = chrono::steady_clock::now();
std::chrono::duration<double> D;

void gt(string s = "")
{
    cerr << s << endl;
    cerr << setprecision(12) << fixed << '\t' << (D = chrono::steady_clock::now() - C).count() << "s" << endl;
    C = chrono::steady_clock::now();
}