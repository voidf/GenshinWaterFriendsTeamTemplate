template <typename T = int>
struct MagicBlossom
{
    T infinity;

    struct Edge
    {
        T to;
        bool used = false, visited = false, matched = false;
        Edge(T _to) : to(_to) {}
    };

    std::vector<T> evenlevel; // 最小偶长度的可选边的边长，没有就是无穷
    std::vector<T> oddlevel;  // 最小奇长度的可选边的边长，没有就是无穷

    std::vector<T> level;  // 上两者的最小值
    std::vector<char> flg; // 0表示outer，即level(x)为偶数，为奇时是1，inner
    std::vector<T> other_level;
    std::vector<T> blossom;
    std::vector<std::set<T>> predecessors;
    std::vector<std::set<T>> anomalies;
    std::vector<std::vector<std::pair<T, T>>> bridges;
    std::vector<char> v_visited;
    std::vector<char> v_matched;
    // std::vector<char> e_visited;
    // std::vector<char> e_used;
    std::vector<std::vector<Edge>> E;

    void init(int siz)
    {
        evenlevel.assign(siz, infinity);
        level.assign(siz, infinity);
        oddlevel.assign(siz, infinity);
        for (auto i : range(1, siz))
            if (not v_matched[i])
                evenlevel[i] = 0;
        other_level.assign(siz, infinity);
        blossom.assign(siz, -1);
        bridges.assign(siz, std::vector<std::pair<T, T>>());
        predecessors.assign(siz, std::set<T>());
        anomalies.assign(siz, std::set<T>());
        v_visited.assign(siz, 0);

        flg.assign(siz, 0);
    }

    void add_edge(T u, T v)
    {
        E[u].emplace_back(v);
        E[v].emplace_back(u);
    }

    MagicBlossom(int siz)
    {
        ++siz;
        E.assign(siz, std::vector<Edge>());
        v_matched.assign(siz, 0);
        memset(&infinity, 0x3f, sizeof(infinity));
    }

    bool bloss_aug(T w1, T w2)
    {
        if (blossom[w1] == blossom[w2] and blossom[w1] != -1)
            return 0;
        
    }

    bool search()
    {
        init(v_matched.size() - 1);
        T i = 0;
        T n = level.size();
        T B = 0;
        bool exitflg = false;
        while (!exitflg)
        {
            exitflg = true;
            if (i & 1)
            {
                for (auto v : range(1, n + 1))
                {
                    if (oddlevel[v] == i)
                    {
                        if (blossom[v] == -1)
                        {
                            for (Edge &e : E[v])
                            {
                                T u = e.to;
                                if (v_matched[u])
                                {
                                    if (oddlevel[u] == i)
                                        bridges[(evenlevel[u] + evenlevel[v]) / 2].emplace_back(u, v);
                                    if (oddlevel[u] == infinity)
                                    {
                                        evenlevel[u] = i + 1;
                                        predecessors[u].clear();
                                        predecessors[u].emplace(v);
                                    }
                                }
                            }
                        }
                        exitflg = false;
                    }
                }
            }
            else
            {
                for (auto v : range(1, n + 1))
                {
                    if (evenlevel == i)
                    {
                        for (Edge &e : E[v])
                        {
                            if (!e.used and !v_matched[e.to])
                            {
                                T u = e.to;
                                if (evenlevel[u] != infinity)
                                {
                                    T tmp = (evenlevel[u] + evenlevel[v]) / 2;
                                    bridges[tmp].emplace_back(u, v);
                                }
                                else
                                {
                                    if (oddlevel[u] == infinity)
                                        oddlevel[u] = i + 1;
                                    if (oddlevel[u] == i + 1)
                                        predecessors[u].emplace(v);
                                    if (oddlevel[u] < i)
                                        anomalies[u].emplace(v);
                                }
                            }
                        }
                        exitflg = false;
                    }
                }
            }
            bool augm_occured = false;
            for (auto &e : bridges[i])
            {
            }
            ++i;
        }
        return 0;
    }
};
