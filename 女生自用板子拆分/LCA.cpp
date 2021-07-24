// #include "Headers.cpp"
// #include "foreach.cpp"

/* 从1到n都可用，0是保留字 5b4026638a0f469f91d26a4ff0dee4bf */
struct LCA
{
    std::vector<std::vector<int>> fa;
    std::vector<int> dep, siz;
    std::vector<std::vector<int>> &E;

    /* 构造函数分配内存，传入边数组 */
    LCA(int _siz, std::vector<std::vector<int>> &_E) : E(_E)
    {
        _siz++;
        fa.assign(_siz, vector<int>(log2int(_siz) + 1, 0));
        dep.assign(_siz, 0);
        siz.assign(_siz, 0);
    }

    void dfs(int x, int from)
    {
        fa[x][0] = from;
        dep[x] = dep[from] + 1;
        siz[x] = 1;
        for (auto i : range(1, log2int(dep[x]) + 1))
            fa[x][i] = fa[fa[x][i - 1]][i - 1];
        for (auto &i : E[x])
            if (i != from)
            {
                dfs(i, x);
                siz[x] += siz[i];
            }
    }

    /* 传入边 */
    void prework(int root)
    {
        // dep[root] = 1;
        dfs(root, 0);
        siz[0] = siz[root];
        // for (auto &i : E[root])
        // dfs(i, root);
    }

    /* LCA查找 */
    int lca(int x, int y)
    {
        if (dep[x] < dep[y])
            swap(x, y);
        while (dep[x] > dep[y])
            x = fa[x][log2int(dep[x] - dep[y])];
        if (x == y)
            return x;
        for (auto k : range(log2int(dep[x]), -1, -1))
            if (fa[x][k] != fa[y][k])
                x = fa[x][k], y = fa[y][k];
        return fa[x][0];
    }

    /* 拿x所在father的子树的节点数 */
    int subtree_size(int x, int father)
    {
        if (x == father)
            return 0;
        for (auto i : range(fa[x].size() - 1, -1, -1))
            x = (dep[fa[x][i]] > dep[father] ? fa[x][i] : x);
        return siz[x];
    }

    /* 判断tobechk是否在from -> to的路径上 */
    bool on_the_way(int from, int to, int tobechk)
    {
        int k = lca(from, to);
        return ((lca(from, tobechk) == tobechk) or (lca(tobechk, to) == tobechk)) and lca(tobechk, k) == k;
    }
};
