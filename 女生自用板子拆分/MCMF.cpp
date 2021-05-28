#include "Headers.cpp"

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
};