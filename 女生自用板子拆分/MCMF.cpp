#include "Headers.cpp"

/* 2021.7.23 完全使用vector版本，SPFA使用SLF优化 */
template <typename T>
struct MCMF // 费用流(Dinic)zkw板子
{           // Based on Dinic (zkw)

    typedef long long LL;
    T INF;
    int N = 1e5 + 5; // 最大点meta参数，要按需改
#define _N 10006
    std::bitset<_N> vis; // 要一起改
    std::vector<T> Dis;
    int s, t;             // 源点，汇点需要外部写入
    std::vector<int> Cur; // 当前弧优化用
    T maxflow, mincost;   // 放最终答案

    struct EdgeContent
    {
        int to;
        T flow;
        T cost;
        int dualEdge;
        EdgeContent(int a, T b, T c, int d) : to(a), flow(b), cost(c), dualEdge(d) {}
    };

    std::vector<std::vector<EdgeContent>> E;

    /* 构造函数，分配内存 */
    MCMF(int n)
    {
        N = n;
        E.assign(n + 1, std::vector<EdgeContent>());
        Dis.assign(n + 1, 0);
        Cur.assign(n + 1, 0);
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
        std::deque<int> Q;
        Q.emplace_back(s);
        // memset(Dis, INF, sizeof(T) * (N + 1));
        Dis.assign(N + 1, INF);
        Dis[s] = 0;
        int k;
        while (!Q.empty())
        {
            k = Q.front();
            Q.pop_front();
            vis.reset(k);
            // for (auto [to, f, w, rev] : E[k])s
            for (auto &i : E[k])
            {
                auto &to = i.to;
                auto &f = i.flow;
                auto &w = i.cost;
                auto &rev = i.dualEdge;
                if (f and Dis[k] + w < Dis[to])
                {
                    Dis[to] = Dis[k] + w;
                    if (!vis.test(to))
                    {
                        if (Q.size() and Dis[Q.front()] > Dis[to])
                        {
                            Q.emplace_front(to);
                        }
                        else
                            Q.emplace_back(to);
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
            auto &to = E[k][i].to;
            auto &f = E[k][i].flow;
            auto &w = E[k][i].cost;
            auto &rev = E[k][i].dualEdge;
            // auto &[to, f, w, rev] = E[k][i];
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
            // memset(Cur, 0, sizeof(int) * (N + 1));
            Cur.assign(N + 1, 0);
            DFS(s, INF);
        }
    }
};

/* 
    2021.10.26 原始对偶版本，除了第一次最短路外之后的最短路都运行在非负图上，可以使用dij 
    但板题表现是始终使用SPFA+SLF最优    
*/
template <typename Cap, typename Cost = Cap>
struct MCMFDUAL // 费用流(Dinic)zkw原始对偶板子
{			// dij开关：first_spfa
			// 非堆开关: #define not_use_heap
			// SLF优化：自己改=_=\\\

	typedef long long LL;
	Cap INF;
	Cost CINF;
	int N;				   // 最大点meta参数，要按需改
						   // #define _N 10006
	std::vector<char> vis; // 要一起改
	std::vector<Cost> Dis;
	int s, t;			  // 源点，汇点需要外部写入
	std::vector<int> Cur; // 当前弧优化用
	Cap maxflow;
	Cost mincost; // 放最终答案
	Cost D;		  
	bool first_spfa;

	struct EdgeContent
	{
		int to;
		Cap flow;
		Cost cost;
		int dualEdge;
		EdgeContent(int a, Cap b, Cost c, int d) : to(a), flow(b), cost(c), dualEdge(d) {}
	};

	std::vector<std::vector<EdgeContent>> E; // 边数组

	/* 构造函数，分配内存 */
	MCMFDUAL(int n)
	{
		N = n;
		D = 0;
		E.assign(n + 1, std::vector<EdgeContent>());
		Dis.assign(n + 1, 0);
		vis.assign(n + 1, 0);
		Cur.assign(n + 1, 0);
		maxflow = mincost = 0;
		memset(&INF, 0x3f, sizeof(INF));
		memset(&CINF, 0x3f, sizeof(CINF));
		first_spfa = true;
	}

	void add(int u, int v, Cap f, Cost w) // 加一条u到v流为f单位费为w的边
	{
		E[u].emplace_back(v, f, w, E[v].size());
		E[v].emplace_back(u, 0, -w, E[u].size() - 1);
	}

	bool SPFA()
	{
		// memset(Dis, INF, sizeof(T) * (N + 1));
		Dis.assign(N + 1, CINF);
		Dis[s] = 0;
		int k;

		// if(first_spfa)
		if (first_spfa)
		{
			std::vector<char> inqueue(N + 1, 0);
			// first_spfa = false;
			// std::queue<int> Q;
			// Q.emplace(s);
			std::deque<int> Q;
			Q.emplace_back(s);
			inqueue[s] = 1;
			while (Q.size())
			{
				k = Q.front();
				// Q.pop();
				Q.pop_front();
				for (auto &i : E[k])
				{
					auto &to = i.to;
					auto &f = i.flow;
					auto &w = i.cost;
					// auto &rev = i.dualEdge;
					if (f and Dis[k] + w < Dis[to])
					{
						Dis[to] = Dis[k] + w;
						if (!inqueue[to])
						{
							if (Q.size() and Dis[Q.front()] >= Dis[to])
								Q.emplace_front(to);
							else
								Q.emplace_back(to);
							Q.emplace(to);
							inqueue[to] = 1;
						}
					}
				}
				inqueue[k] = 0;
			}
		}
		else
		{
			std::vector<char> dvis(N + 1, 0);
#ifndef not_use_heap
			struct elem
			{
				int x;
				Cost k;
				bool operator<(const elem &b) const { return k > b.k; };
				elem(int px, Cost key) : x(px), k(key) {}
			};
			std::priority_queue<elem> Q;
			Q.emplace(s, Dis[s]);
			while (Q.size())
			{
				k = Q.top().x;
				dvis[k] = 1;
				Q.pop();
				for (auto &i : E[k])
				{
					auto &to = i.to;
					auto &f = i.flow;
					auto &w = i.cost;
					// auto &rev = i.dualEdge;
					if (f and Dis[k] + w < Dis[to])
					{
						Dis[to] = Dis[k] + w;
						if (!dvis[to])
							Q.emplace(to, Dis[to]);
					}
				}
			}
#else
			// 非堆

			int ato = N + 1;
			while (ato--)
			{
				int k = -1;
				for (int i = 0; i <= N; ++i)
					if (!dvis[i] and (k == -1 or Dis[i] < Dis[k]))
						k = i;
				dvis[k] = 1;
				for (auto &i : E[k])
				{
					auto &to = i.to;
					auto &f = i.flow;
					auto &w = i.cost;
					if (f and Dis[k] + w < Dis[to])
						Dis[to] = Dis[k] + w;
				}
			}
#endif
		}
		for (int i = 0; i <= N; ++i)
			for (auto &j : E[i])
				j.cost -= Dis[j.to] - Dis[i];
		D += Dis[t];
		return Dis[t] != CINF;
	}
	Cap DFS(int k, Cap flow)
	{
		if (k == t)
		{
			maxflow += flow;
			mincost += D * flow;
			return flow;
		}
		Cap sum = 0;
		vis[k] = 1;
		for (auto &i = Cur[k]; i < E[k].size(); ++i)
		{
			auto &to = E[k][i].to;
			auto &f = E[k][i].flow;
			auto &w = E[k][i].cost;
			auto &rev = E[k][i].dualEdge;
			// auto &[to, f, w, rev] = E[k][i];
			if (!vis[to] and f and !w)
			{
				Cap p = DFS(to, std::min(flow - sum, f));
				sum += p;
				f -= p;
				E[to][rev].flow += p;
				if (sum == flow)
					break;
			}
		}
		return sum;
	}

	void Dinic() // 入口
	{
		while (SPFA())
		{
			do
			{
				vis.assign(N + 1, 0);
				Cur.assign(N + 1, 0);
			} while (DFS(s, INF));
		}
	}
};
