#include "Headers.cpp"
#include "foreach.cpp"

/* Tarjan求割点 67097c1850be2e3cfed344d3d4e1142d */
struct Tarjan
{
    std::vector<int> DFN, LOW;
    std::vector<int> belongs;
    std::vector<int> DFS_from; // 记父亲节点
    std::vector<std::set<int>> &E; // 根据实际情况选择set还是vector
    std::vector<char> in_stack;
    std::stack<int> stk;
    std::vector<int> changed; // 被缩点的点
    int ts;
    std::set<int> cut; // 割点
    int N;
    int remaining_point_ctr;

    /* 构造函数确定边引用 */
    Tarjan(int _siz, std::vector<std::set<int>> &_E) : E(_E), N(_siz + 1) {}

    /* 类并查集路径压缩寻找SCC代表节点 */
    inline int chk_belongs(int x)
    {
        if (belongs[x] == x)
            return x;
        else
            return belongs[x] = chk_belongs(belongs[x]);
    }
    /* 为多次运行准备的初始化函数 */
    void init()
    {
        DFN.assign(N, 0);
        LOW.assign(N, 0);
        DFS_from.assign(N, 0);
        belongs.assign(N, 0);
        in_stack.assign(N, 0);
        ts = 0;
    }

    /* 入口 */
    void run()
    {
        init();
        for (auto i : range(1, DFN.size()))
            if (!DFN[i])
                tarjan(i, i);
        remaining_point_ctr = N - 1 - changed.size();
    }
    /* 内部用，x==f时表示本节点为根节点 */
    inline void tarjan(int x, int f) // 本意是处理无向图
    {
        DFS_from[x] = f;
        DFN[x] = LOW[x] = ++ts;
        in_stack[x] = 1;
        stk.push(x);
        if (x == f) // 本节点为根
        {
            set<int> realson;
            for (auto &i : E[x])
            {
                if (!DFN[i])
                {
                    tarjan(i, x);
                    LOW[x] = min(LOW[x], LOW[i]);
                    if (realson.size() < 2)
                        realson.insert(LOW[i]);
                }
                else if (in_stack[i])
                    LOW[x] = min(LOW[x], DFN[i]);
            }
            if (realson.size() >= 2)
                cut.insert(x);
        }
        else
        {
            for (auto &i : E[x])
            {
                // if (i != f) // 无向图这么写
                if (1)
                {
                    if (!DFN[i])
                    {
                        tarjan(i, x);
                        LOW[x] = min(LOW[x], LOW[i]);
                        if (LOW[i] >= DFN[x])
                            cut.insert(x);
                    }
                    else if (in_stack[i])
                        LOW[x] = min(LOW[x], DFN[i]);
                }
            }
        }
        if (DFN[x] == LOW[x])
        {
            while (stk.size())
            {
                int tp = stk.top();
                in_stack[tp] = 0;
                stk.pop();
                belongs[tp] = x;
                if (x != tp)
                    changed.push_back(tp);
                if (tp == x)
                    break;
            }
        }
    }
    /* 注意这步还没有完全更新边，遍历边时必须使用auto i: E[x], belongs[i]，或使用下面的合并边 */
    void do_merge()
    {
        for (auto i : changed)
        {
            for (auto j : E[i])
            {
                int fi = chk_belongs(i);
                int fj = chk_belongs(j);
                if (fi != fj)
                    E[fi].emplace(fj);
            }
            E[i].clear();
        }
        changed.clear();
    }

    /* 主动将合并后的边整理并去重，多次使用可能TLE */
    void handle_merged_edge()
    {
        for (auto i : range(1, N))
        {
            if (E[i].size())
            {
                update_point(i);
            }
        }
    }

    inline void update_point(int x)
    {
        set<int> tmpe;
        for (auto j : E[x])
        {
            int fj = chk_belongs(j);
            if (fj != x)
                tmpe.emplace(fj);
        }
        swap(E[x], tmpe);
    }

    /* 仅加一条边的缩点，在已经跑过上面的缩点之后使用，为了保证复杂度实际上只维护了一个并查集 */
    void single_edge_SCC(int u, int v)
    {
        u = chk_belongs(u);
        v = chk_belongs(v);
        int father;
        while (u != v)
        {
            if (DFN[u] < DFN[v])
                swap(u, v);
            changed.push_back(u);
            u = chk_belongs(DFS_from[u]);
        }
        for (auto i : changed)
        {
            belongs[i] = u;
            // remaining_points.erase(i);
        }
        remaining_point_ctr -= changed.size();
        // do_merge();
        changed.clear();
    }
};

/*
附一份C++98的板子 
https://vjudge.net/problem/POJ-3694
struct Tarjan
{
    std::vector<int> DFN, LOW;
    std::vector<int> belongs;
    std::vector<int> DFS_from;
    std::vector<std::vector<int>> &E;
    std::set<int> remaining_points;
    std::vector<char> in_stack;
    std::stack<int> stk;
    std::vector<int> changed; // 被缩点的点
    int ts;
    std::set<int> cut; // 割点
    int N;
    int remaining_point_ctr;

    inline int chk_belongs(int x)
    {
        if (belongs[x] == x)
            return x;
        else
            return belongs[x] = chk_belongs(belongs[x]);
    }

    // 构造函数确定边引用
    Tarjan(int _siz, std::vector<std::vector<int>> &_E) : E(_E), N(_siz + 1)
    {
        // for (auto i : range(1, _siz + 1))
        // for (int i = 1; i <= _siz; ++i)
        // remaining_points.insert(i);
    }

    // 为多次运行准备的初始化函数
    void init()
    {
        DFN.assign(N, 0);
        LOW.assign(N, 0);
        DFS_from.assign(N, 0);
        belongs.assign(N, 0);
        in_stack.assign(N, 0);
        ts = 0;
    }

    // 入口
    void run()
    {
        init();
        // for (auto i : range(1, DFN.size()))
        for (int i = 1; i < DFN.size(); ++i)
            if (!DFN[i])
                tarjan(i, i);
        remaining_point_ctr = N - 1 - changed.size();
    }
    // 内部用，x==f时表示本节点为根节点
    inline void tarjan(int x, int f) // 本意是处理无向图
    {
        DFN[x] = LOW[x] = ++ts;
        DFS_from[x] = f;
        in_stack[x] = 1;
        stk.push(x);
        if (x == f) // 本节点为根
        {
            // set<int> realson;
            // for (auto &i : E[x])
            for (std::vector<int>::iterator ii = E[x].begin(); ii != E[x].end(); ii++)
            {
                int i = *ii;
                if (!DFN[i])
                {
                    tarjan(i, x);
                    LOW[x] = min(LOW[x], LOW[i]);
                    // if (realson.size() < 2)
                        // realson.insert(LOW[i]);
                }
                else if (in_stack[i])
                    LOW[x] = min(LOW[x], DFN[i]);
            }
            // if (realson.size() >= 2)
            // cut.insert(x);
        }
        else
        {
            // for (auto &i : E[x])
            for (std::vector<int>::iterator ii = E[x].begin(); ii != E[x].end(); ii++)
            {
                int i = *ii;
                if (i != f) // 无向图这么写
                // if (1)
                {
                    if (!DFN[i])
                    {
                        tarjan(i, x);
                        LOW[x] = min(LOW[x], LOW[i]);
                        // if (LOW[i] >= DFN[x])
                        // cut.insert(x);
                    }
                    else if (in_stack[i])
                        LOW[x] = min(LOW[x], DFN[i]);
                }
            }
        }
        if (DFN[x] == LOW[x])
        {
            while (stk.size())
            {
                int tp = stk.top();
                in_stack[tp] = 0;
                stk.pop();
                belongs[tp] = x;
                if (x != tp)
                    changed.push_back(tp);
                if (tp == x)
                    break;
            }
        }
    }
    // 注意这步还没有完全更新边，遍历边时必须使用auto i: E[x], belongs[i]，或使用下面的合并边
    void do_merge()
    {
        // for (auto i : changed)
        for (std::vector<int>::iterator ii = changed.begin(); ii != changed.end(); ii++)
        {
            int i = *ii;
            // for (auto j : E[i])
            for (std::vector<int>::iterator jj = E[i].begin(); jj != E[i].end(); jj++)
            {
                int j = *jj;
                int fi = chk_belongs(i);
                int fj = chk_belongs(j);
                if (fi != fj)
                    E[fi].push_back(fj);
            }
            E[i].clear();
            remaining_points.erase(i);
        }
        changed.clear();
    }

    inline void update_point(int x)
    {
        set<int> tmpe;
        // for (auto j : E[x])
        for (std::vector<int>::iterator jj = E[x].begin(); jj != E[x].end(); jj++)
        {
            int j = *jj;
            if (belongs[j] != x)
                tmpe.insert(belongs[j]);
            // tmpe.emplace(belongs[j]);
        }
        // swap(E[x], tmpe);
    }

    // 将合并后的边整理并去重
    void handle_merged_edge()
    {
        // for (auto i : remaining_points)
        for (std::set<int>::iterator jj = remaining_points.begin(); jj != remaining_points.end(); jj++)
        {
            int i = *jj;
            if (E[i].size())
            {
                update_point(i);
            }
        }
    }
    // 仅加一条边的缩点，在已经跑过上面的缩点之后使用，实际上只维护了一个并查集
    void single_edge_SCC(int u, int v)
    {
        u = chk_belongs(u);
        v = chk_belongs(v);
        std::vector<int> tobeupdate;
        int father;
        while (u != v)
        {
            if (DFN[u] < DFN[v])
                swap(u, v);
            changed.push_back(u);
            u = chk_belongs(DFS_from[u]);
        }
        // for (auto i : changed)
        for (std::vector<int>::iterator ii = changed.begin(); ii != changed.end(); ii++)
        {
            int i = *ii;
            belongs[i] = u;
            // remaining_points.erase(i);
        }
        remaining_point_ctr -= changed.size();
        // do_merge();
        changed.clear();
    }
};

// LL n, u, v;
int n, m;
void solve()
{
    // qr(n);
    // qr(m);

    vector<vector<int>> E(n + 1, vector<int>());
    // for (auto i : range(m))
    for (int i = 0; i < m; ++i)
    {
        int u, v;
        qr(u);
        qr(v);
        E[u].push_back(v);
        E[v].push_back(u);
    }
    Tarjan tj(n, E);
    // for(auto i)
    tj.run();
    tj.changed.clear();
    // int rempoint = n - tj.changed.size();
    // tj.do_merge();
    // tj.handle_merged_edge();

    qr(m);
    // for (auto i : range(m))
    for (int i = 0; i < m; ++i)
    {
        int u, v;
        qr(u);
        qr(v);
        tj.single_edge_SCC(u, v);
        printf("%d\n", tj.remaining_point_ctr - 1);
        // cout << tj.remaining_points.size() - 1 << endl;
        // E[tj.belongs[u]]
        // set<int> merged;
        // set<int> involved;
    }
}
*/