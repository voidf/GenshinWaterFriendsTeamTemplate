#pragma GCC optimize(1)
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
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