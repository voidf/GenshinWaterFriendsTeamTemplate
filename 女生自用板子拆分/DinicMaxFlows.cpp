#include "Headers.cpp"




#define DINIC 520010

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