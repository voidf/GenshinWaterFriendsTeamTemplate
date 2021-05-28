#include "Headers.cpp"



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
}