
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

template <typename T>
struct ODT2 // 第二版珂朵莉树，解决区间长度计数最值问题，可以单点修改
{
    struct Node
    {
        int l;
        mutable int r;
        mutable T v;
        Node(int _l) : l(_l), r(_l) {}
        Node(int _l, int _r) : l(_l), r(_r) {}
        Node(int _l, int _r, const T &_v) : l(_l), r(_r), v(_v) {}
        inline bool operator<(const Node &o) const { return l < o.l; }
    };
    using ND = Node;
    multiset<int> S; // 维护长度计数

    void purge(const Node &x) { S.erase(S.find(x.r - x.l + 1)); }
    void install(const Node &x) { S.insert(x.r - x.l + 1); }
    std::set<ND> D;
    typename std::set<ND>::iterator push(const Node &x)
    {
        install(x);
        return D.emplace(x).first;
    }
    void pop(typename std::set<ND>::iterator x)
    {
        purge(*x);
        D.erase(x);
    }

    /* 将包含x的区间[l, r]分裂为[l, x)和[x, r]并返回指向后者的迭代器 
        不需要维护区间个数计数的话应该注释掉S相关操作
    */
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
        purge(*it); //
        D.erase(it);
        auto ttt = D.emplace(ND(tmp.l, x - 1, tmp.v)).first;
        install(*ttt); //
        auto insted = D.insert(ND(x, tmp.r, tmp.v)).first;
        install(*insted); //
        return insted;
    }
    typename std::set<ND>::iterator merge_back(typename std::set<ND>::iterator x)
    {
        auto n = x;
        ++n;
        if (n == D.end() or n->v != x->v)
            return x;
        auto tmp = *x;
        tmp.r = n->r;
        pop(x);
        pop(n);
        return push(tmp);
    }
    typename std::set<ND>::iterator merge_front(typename std::set<ND>::iterator x)
    {
        if (x == D.begin())
            return x;
        auto p = x;
        --p;
        if (p->v != x->v)
            return x;
        auto tmp = *x;
        tmp.l = p->l;
        pop(x);
        pop(p);
        return push(tmp);
    }
    typename std::set<ND>::iterator update_and_merge(int x, const T &v)
    {
        auto p = split(x);
        if (p->l != p->r)
        {
            ND tmp(*p);
            pop(p);
            ++tmp.l;
            push(tmp);
            tmp.l = tmp.r = x;
            p = push(tmp);
        }
        p->v = v; // lift操作
        p = merge_back(p);
        p = merge_front(p);
        return p;
    }
    void assign(int l, int r, const T &v)
    {
        auto itr = split(r + 1), itl = split(l);
        D.erase(itl, itr);
        D.emplace(l, r, v);
    }
};