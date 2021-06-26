template <typename TYP>
struct PersistentSengmentTree
{
    static const int size_elapsed = 9000000; // 因为需要静态分配，在这里指定预估最大大小

    static inline int mid(int lower, int upper) { return (lower + upper) >> 1; };

    int cursiz = 0;
    int l_bound_cache, r_bound_cache;
    struct Node
    {
        TYP meta;

        TYP lazy;

        int l, r;
        Node() : meta(), l(-1), r(-1) {}
        void at_build()
        {
            // memset(meta, 0, sizeof(meta));
            meta = 0;
            lazy = 0;
            l = -1;
            r = -1;
        }
    } nodes[size_elapsed];

    int headers[size_elapsed];
    int h_pointer = 0;

    void init()
    {
        h_pointer = 0;
        cursiz = 0;
    }

    // std::vector<Node> nodes; // 动态开点的选项
    // PersistentSengmentTree() : cursiz(0) { nodes.resize(size_elapsed); }

    inline int _build(int l_bound, int r_bound)
    {
        int cur_num = cursiz++;
        Node &me = nodes[cur_num];
        me.at_build();
        if (r_bound > l_bound)
        {
            int m = mid(l_bound, r_bound);
            me.l = _build(l_bound, m);
            me.r = _build(m + 1, r_bound);
        }
        return cur_num;
    }

    void build(int _n)
    {
        headers[h_pointer++] = _build(1, _n);
        l_bound_cache = 1 - 1;
        r_bound_cache = _n - 1;
    }

    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &&updval)
    {
        int cur_num = cursiz++;
        nodes[cur_num] = nodes[before];
        Node &me = nodes[cur_num];
        // 这里改更新策略
        me.meta += updval;
        //
        // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
        if (l_bound < r_bound)
        {
            int m = mid(l_bound, r_bound);
            if (pos <= m) // 值域线段树，落在哪边就往哪边走
                me.l = _update(l_bound, m, pos, me.l, updval);
            else
                me.r = _update(m + 1, r_bound, pos, me.r, updval);
        }
        return cur_num;
    }
    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &updval) { return _update(l_bound, r_bound, pos, before, std::move(updval)); }

    inline void pushdown(Node &ME, int lb, int rb)
    {
        int m = mid(lb, rb);
        if (ME.lazy)
        {
            nodes[ME.l].meta += (m - lb + 1) * ME.lazy;
            nodes[ME.r].meta += (rb - m + 1) * ME.lazy;
            nodes[ME.l].lazy += ME.lazy;
            nodes[ME.r].lazy += ME.lazy;
            ME.lazy = 0;
        }
    }

    inline int _updateR(int l_bound, int r_bound, int ql, int qr, int before, TYP &&updval)
    {
        int cur_num = cursiz++;
        nodes[cur_num] = nodes[before];
        Node &me = nodes[cur_num];
        // 这里改更新策略
        // me.meta += updval;
        if (ql <= l_bound and r_bound <= qr)
        {
            me.lazy += updval;
            me.meta += (updval) * (r_bound - l_bound + 1);
            return cur_num;
        }

        //
        // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
        pushdown(me, l_bound, r_bound);
        int m = mid(l_bound, r_bound);

        if (ql <= m)
            me.l = _updateR(l_bound, m, ql, qr, me.l, updval);
        if (m + 1 <= qr)
            me.r = _updateR(m + 1, r_bound, ql, qr, me.r, updval);
        me.lazy = 0;
        me.meta = nodes[me.l].meta + nodes[me.r].meta;
        return cur_num;
    }
    inline int _updateR(int l_bound, int r_bound, int ql, int qr, int before, TYP &updval) { return _updateR(l_bound, r_bound, ql, qr, before, std::move(updval)); }

    void update(int pos, TYP &&updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }
    void update(int pos, TYP &updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }

    void updateR(int l, int r, TYP &updval)
    {
        headers[h_pointer] = _updateR(l_bound_cache, r_bound_cache, l, r, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }
    void updateR(int l, int r, TYP &&updval)
    {
        headers[h_pointer] = _updateR(l_bound_cache, r_bound_cache, l, r, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }

    /*查询的rank是排名，返回的是离散化的排好序的序列的下标，查询函数根据业务需求改*/
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &&rank)
    {
        Node &u = nodes[front_node];
        Node &v = nodes[back_node];
        if (l_bound >= r_bound)
            return l_bound;
        TYP lx = nodes[v.l].meta - nodes[u.l].meta;
        // TYP rx = nodes[v.r].meta - nodes[u.r].meta;
        // TYP lx = nodes[v.l].meta;
        // cerr << "vlmeta:" << nodes[v.l].meta << "\tulmeta:" << nodes[u.l].meta << endl;
        // cerr << "vrmeta:" << nodes[v.r].meta << "\turmeta:" << nodes[u.r].meta << endl;
        int m = mid(l_bound, r_bound);
        if (lx >= rank)
            return _query(l_bound, m, u.l, v.l, rank);
        else //if (2 * rx > rank)
            return _query(m + 1, r_bound, u.r, v.r, rank - lx);
        // return 0;
    }
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &rank) { return _query(l_bound, r_bound, front_node, back_node, std::move(rank)); }

    int query(int l, int r, TYP &&k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }
    int query(int l, int r, TYP &k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }

    TYP legacy_query(int l_bound, int r_bound, int ql, int qr, int cur)
    {
        Node &me = nodes[cur];
        if (l_bound >= ql and r_bound <= qr)
            return me.meta;
        pushdown(me, l_bound, r_bound);
        TYP res = 0;
        int m = mid(l_bound, r_bound);
        if (ql <= m)
            res += legacy_query(l_bound, m, ql, qr, me.l);
        if (qr >= m + 1)
            res += legacy_query(m + 1, r_bound, ql, qr, me.r);
        return res;
    }

    TYP queryR(int l, int r, int begin_time, int end_time)
    {
        TYP r1 = legacy_query(l_bound_cache, r_bound_cache, l, r, headers[begin_time - 1]);
        TYP r2 = legacy_query(l_bound_cache, r_bound_cache, l, r, headers[end_time]);
        return r2 - r1;
    }
};

template <typename TYP>
struct PersistentSengmentTreeR
{
    static const int size_elapsed = 8000000; // 因为需要静态分配，在这里指定预估最大大小

    static inline int mid(int lower, int upper) { return (lower + upper) >> 1; };

    int cursiz = 0;
    int l_bound_cache, r_bound_cache;
    struct Node
    {
        TYP meta;

        TYP lazy;

        int l, r;
        Node() : meta(), l(-1), r(-1) {}
        void at_build()
        {
            // memset(meta, 0, sizeof(meta));
            meta = 0;
            lazy = 0;
            l = -1;
            r = -1;
        }
    } nodes[size_elapsed];

    int headers[size_elapsed];
    int h_pointer = 0;

    void init()
    {
        h_pointer = 0;
        cursiz = 0;
    }

    // std::vector<Node> nodes; // 动态开点的选项
    // PersistentSengmentTree() : cursiz(0) { nodes.resize(size_elapsed); }

    inline int _build(int l_bound, int r_bound)
    {
        int cur_num = cursiz++;
        Node &me = nodes[cur_num];
        me.at_build();
        if (r_bound > l_bound)
        {
            int m = mid(l_bound, r_bound);
            me.l = _build(l_bound, m);
            me.r = _build(m + 1, r_bound);
        }
        return cur_num;
    }

    void build(int _n)
    {
        headers[h_pointer++] = _build(1, _n);
        l_bound_cache = 1 - 1;
        r_bound_cache = _n - 1;
    }

    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &&updval)
    {
        int cur_num = cursiz++;
        nodes[cur_num] = nodes[before];
        Node &me = nodes[cur_num];
        // 这里改更新策略
        me.meta += updval;
        //
        // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
        if (l_bound < r_bound)
        {
            int m = mid(l_bound, r_bound);
            if (pos <= m) // 值域线段树，落在哪边就往哪边走
                me.l = _update(l_bound, m, pos, me.l, updval);
            else
                me.r = _update(m + 1, r_bound, pos, me.r, updval);
        }
        return cur_num;
    }
    inline int _update(int l_bound, int r_bound, int pos, int before, TYP &updval) { return _update(l_bound, r_bound, pos, before, std::move(updval)); }

    // inline void pushdown(Node &ME, int lb, int rb)
    // {
    //     int m = mid(lb, rb);
    //     if (ME.lazy)
    //     {
    //         nodes[ME.l].meta += (m - lb + 1) * ME.lazy;
    //         nodes[ME.r].meta += (rb - m) * ME.lazy;
    //         nodes[ME.l].lazy += ME.lazy;
    //         nodes[ME.r].lazy += ME.lazy;
    //         ME.lazy = 0;
    //     }
    // }

    inline int _updateR(int l_bound, int r_bound, int ql, int qr, TYP sulazy, int before, TYP &&updval)
    {
        int cur_num = cursiz++;
        nodes[cur_num] = nodes[before];
        Node &me = nodes[cur_num];
        // 这里改更新策略
        // me.meta += updval;
        if (ql <= l_bound and r_bound <= qr)
        {
            me.lazy += updval + sulazy;
            me.meta += (updval + sulazy) * (r_bound - l_bound + 1);
            // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
            return cur_num;
        }

        //
        // pushdown(me, l_bound, r_bound);
        sulazy += me.lazy;
        int m = mid(l_bound, r_bound);

        if (ql <= m)
            me.l = _updateR(l_bound, m, ql, qr, sulazy, me.l, updval);
        if (m + 1 <= qr)
            me.r = _updateR(m + 1, r_bound, ql, qr, sulazy, me.r, updval);
        me.lazy = 0;
        me.meta = nodes[me.l].meta + nodes[me.r].meta;
        // cerr << "[" << l_bound << ", " << r_bound << "]: " << me.meta << "\t+" << cur_num << endl;
        return cur_num;
    }
    inline int _updateR(int l_bound, int r_bound, int ql, int qr, TYP sulazy, int before, TYP &updval) { return _updateR(l_bound, r_bound, ql, qr, sulazy, before, std::move(updval)); }

    void update(int pos, TYP &&updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }
    void update(int pos, TYP &updval)
    {
        headers[h_pointer] = _update(l_bound_cache, r_bound_cache, pos, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }

    void updateR(int l, int r, TYP &updval)
    {
        headers[h_pointer] = _updateR(l_bound_cache, r_bound_cache, l, r, 0, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }
    void updateR(int l, int r, TYP &&updval)
    {
        headers[h_pointer] = _updateR(l_bound_cache, r_bound_cache, l, r, 0, headers[h_pointer - 1], std::move(updval));
        h_pointer++;
    }

    /*查询的rank是排名，返回的是离散化的排好序的序列的下标，查询函数根据业务需求改*/
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &&rank)
    {
        Node &u = nodes[front_node];
        Node &v = nodes[back_node];
        if (l_bound >= r_bound)
            return l_bound;
        TYP lx = nodes[v.l].meta - nodes[u.l].meta;
        // TYP rx = nodes[v.r].meta - nodes[u.r].meta;
        // TYP lx = nodes[v.l].meta;
        // cerr << "vlmeta:" << nodes[v.l].meta << "\tulmeta:" << nodes[u.l].meta << endl;
        // cerr << "vrmeta:" << nodes[v.r].meta << "\turmeta:" << nodes[u.r].meta << endl;
        int m = mid(l_bound, r_bound);
        if (lx >= rank)
            return _query(l_bound, m, u.l, v.l, rank);
        else //if (2 * rx > rank)
            return _query(m + 1, r_bound, u.r, v.r, rank - lx);
        // return 0;
    }
    inline int _query(int l_bound, int r_bound, int front_node, int back_node, TYP &rank) { return _query(l_bound, r_bound, front_node, back_node, std::move(rank)); }

    int query(int l, int r, TYP &&k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }
    int query(int l, int r, TYP &k)
    {
        return _query(l_bound_cache, r_bound_cache, headers[l - 1], headers[r], k);
    }

    TYP legacy_query(int l_bound, int r_bound, int ql, int qr, int cur, TYP sulazy)
    {
        Node &me = nodes[cur];
        if (l_bound >= ql and r_bound <= qr)
            return me.meta + sulazy * (r_bound - l_bound + 1);
        // pushdown(me, l_bound, r_bound);
        sulazy += me.lazy;
        TYP res = 0;
        int m = mid(l_bound, r_bound);
        if (ql <= m)
            res += legacy_query(l_bound, m, ql, qr, me.l, sulazy);
        if (qr >= m + 1)
            res += legacy_query(m + 1, r_bound, ql, qr, me.r, sulazy);
        return res;
    }

    TYP queryR(int l, int r, int begin_time, int end_time)
    {
        TYP r1 = legacy_query(l_bound_cache, r_bound_cache, l, r, headers[begin_time - 1], 0);
        TYP r2 = legacy_query(l_bound_cache, r_bound_cache, l, r, headers[end_time], 0);
        return r2 - r1;
    }
};
