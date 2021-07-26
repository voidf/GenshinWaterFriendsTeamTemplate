#include "Headers.cpp"
#include "foreach.cpp"

/* Tarjan求割点 */
struct Tarjan
{
    std::vector<int> DFN, LOW;
    std::vector<int> belongs;
    std::vector<std::set<int>> &E;
    std::vector<char> in_stack;
    std::stack<int> stk;
    std::vector<int> changed; // 被缩点的点
    int ts;
    std::set<int> cut; // 割点
    int N;

    /* 构造函数确定边引用 */
    Tarjan(int _siz, std::vector<std::set<int>> &_E) : E(_E), N(_siz + 1) {}

    /* 为多次运行准备的初始化函数 */
    void init()
    {
        DFN.assign(N, 0);
        LOW.assign(N, 0);
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
    }
    /* 内部用，x==f时表示本节点为根节点 */
    inline void tarjan(int x, int f) // 本意是处理无向图
    {
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
                E[belongs[i]].emplace(belongs[j]);
            }
            E[i].clear();
        }
        changed.clear();
    }
    /* 将合并后的边整理并去重 */
    void handle_merged_edge()
    {
        for (auto i : range(1, N))
        {
            if (E[i].size())
            {
                set<int> tmpe;
                for (auto j : E[i])
                    if (belongs[j] != i)
                        tmpe.emplace(belongs[j]);
                swap(E[i], tmpe);
            }
        }
    }
};