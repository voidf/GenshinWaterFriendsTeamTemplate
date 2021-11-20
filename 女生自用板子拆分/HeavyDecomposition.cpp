#include "Headers.cpp"


struct HeavyDecomposition
{
    // 深度，父亲，重儿子，映射到数据结构上的编号(dfs序)，以该点为根子树大小，链顶编号
    std::vector<int> dep, fa, hson, nid, sz, top;
    const std::vector<std::vector<int>> &E; // 引用的边数组
    int bp = 0;                             // 映射起点偏移量，若从1开始请设为1
    /* s:问题规模，_E:树的边数组 */
    HeavyDecomposition(int s,
                       const std::vector<std::vector<int>> &_E)
        : dep(s + 1),
          fa(s + 1),
          hson(s + 1, -1),
          nid(s + 1),
          sz(s + 1),
          top(s + 1),
          E(_E) {}
    /* 处理深度，记父亲，子树大小，传入d是当前深度 */
    void dfs1(int x, int f, int d)
    {
        dep[x] = d;
        fa[x] = f;
        sz[x] = 1;
        int mxsonsize = -1;
        for (auto i : E[x])
            if (i != f)
            {
                dfs1(i, x, d + 1);
                sz[x] += sz[i];
                if (sz[i] > mxsonsize)
                    mxsonsize = sz[i], hson[x] = i;
            }
    }

    void dfs2(int x, int tp)
    {
        top[x] = tp;
        nid[x] = bp++;
        if (hson[x] == -1)
            return;
        dfs2(hson[x], tp);
        for (auto i : E[x])
            if (fa[x] != i && hson[x] != i)
                dfs2(i, i);
    }
    /* 预处理入口，处理完毕后直接访问nid[x]即可获得x的dfs序 */
    inline void prework(int root)
    {
        dfs1(root, root, 0);
        dfs2(root, root);
    }
    /* 获得树上u->v简单路径在序列上的区间映射，解析子树区间请直接用(nid[u], nid[u]+sz[u]-1) */
    inline std::vector<std::pair<int, int>> resolve_path(int u, int v)
    {
        std::vector<std::pair<int, int>> R;
        while (top[u] != top[v])
        {
            if (dep[top[u]] < dep[top[v]]) // 令u链顶为深度大的点
                swap(u, v);
            R.emplace_back(nid[top[u]], nid[u]); // 计入u的链顶到u的区间，然后令u向上爬
            u = fa[top[u]];
        }
        // 此时u,v top相同，在同一条链上，令u更深，添加[v, u]区间
        if (dep[u] < dep[v])
            swap(u, v);
        R.emplace_back(nid[v], nid[u]);
        return R;
    }
};