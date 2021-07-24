#include "Headers.cpp"
#include "foreach.cpp"
/* 从1到n都可用，0是保留字 */
struct LCA
{
    std::vector<std::vector<int>> fa;
    std::vector<int> dep;
    std::vector<std::vector<int>> &E;
    /* 构造函数分配内存，传入边数组 */
    LCA(int siz, std::vector<std::vector<int>> &_E) : E(_E)
    {
        siz++;
        fa.assign(siz, vector<int>(log2int(siz) + 1, 0));
        dep.assign(siz, 0);
    }

    void dfs(int x, int from)
    {
        fa[x][0] = from;
        dep[x] = dep[from] + 1;
        for (auto i : range(1, log2int(dep[x]) + 1))
            fa[x][i] = fa[fa[x][i - 1]][i - 1];
        for (auto &i : E[x])
            if (i != from)
                dfs(i, x);
    }

    /* 传入边 */
    void prework(int root)
    {
        dep[root] = 1;
        for (auto &i : E[root])
            dfs(i, root);
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
};

