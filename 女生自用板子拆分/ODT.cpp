
template <typename T>
struct ODT
{
    struct Node
    {
        int l, r;
        mutable T v;
        Node(int _l) : l(_l), r(_l) {}
        Node(int _l, int _r) : l(_l), r(_r) {}
        Node(int _l, int _r, const T &_v) : l(_l), r(_r), v(_v) {}
        inline bool operator<(const Node &o) const { return l < o.l; }
    };
    using ND = Node;
    std::set<ND> D;
    /* 将包含x的区间[l, r]分裂为[l, x)和[x, r]并返回指向后者的迭代器 */
    typename std::set<ND>::iterator split(int x)
    {
        auto ub = D.upper_bound(ND(x));
        if (ub == D.begin())
            return ub;
        auto it = ub;
        --it;

        if (it->l == x)
            return it;
        else if (it->r < x)
            return ub;
        ND tmp(*it);
        D.erase(it);
        D.emplace(ND(tmp.l, x - 1, tmp.v));
        return D.insert(ND(x, tmp.r, tmp.v)).first;
    }
    void assign(int l, int r, const T &v)
    {
        auto itr = split(r + 1), itl = split(l);
        D.erase(itl, itr);
        D.emplace(l, r, v);
    }
};