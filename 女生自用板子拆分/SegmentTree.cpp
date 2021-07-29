
namespace Tree
{
    template <typename T>
    struct _iNode
    {
        T lazy_add;
        T sum_content;
        T lazy_mul;
        // T max_content;
        T min_content;
        _iNode() : lazy_add(0), sum_content(0), lazy_mul(1), min_content(0x3f3f3f3f) {}
    };

    template <typename T>
    struct SegmentTree
    {
        using _Node = _iNode<T>;
        int len;       // 线段树实际节点数
        int valid_len; // 原有效数据长度
        std::vector<_Node> _D;
        // template <typename AllocationPlaceType = void>
        SegmentTree(int length, void *arr = nullptr) // 构造函数只分配内存
        {
            valid_len = length;
            len = 1 << 1 + (int)ceil(log2(length));
            _D.resize(len);
        }

        void show()
        {
            std::cout << '[';
            for (_Node *i = _D.begin(); i != _D.end(); ++i)
                std::cout << i->sum_content << ",]"[i == _D.end() - 1] << " \n"[i == _D.end() - 1];
        }

        static int mid(int l, int r) { return l + r >> 1; }

        void update_mul(int l,
                        int r,
                        T v,
                        int node_l,
                        int node_r,
                        int x)
        {
            if (l <= node_l and node_r <= r)
            {
                _D[x].lazy_add *= v;
                _D[x].sum_content *= v;
                _D[x].lazy_mul *= v;
                _D[x].min_content *= v;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    update_mul(l, r, v, node_l, mi, x << 1);
                if (r > mi)
                    update_mul(l, r, v, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void update_add(int l,
                        int r,
                        T v,
                        int node_l,
                        int node_r,
                        int x)
        {
            if (l <= node_l and node_r <= r)
            {
                LL my_length = node_r - node_l + 1;
                _D[x].lazy_add += v;
                _D[x].sum_content += my_length * v;
                _D[x].min_content += v;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    update_add(l, r, v, node_l, mi, x << 1);
                if (r > mi)
                    update_add(l, r, v, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void range_mul(int l, int r, T v)
        {
            update_mul(l, r, v, 1, valid_len, 1);
        }

        void range_add(int l, int r, T v)
        {
            update_add(l, r, v, 1, valid_len, 1);
        }

        inline void maintain(int i)
        {
            int l = i << 1;
            int r = l | 1;
            _D[i].sum_content = (_D[l].sum_content + _D[r].sum_content);
            _D[i].min_content = min(_D[l].min_content, _D[r].min_content);
        }

        inline void push_down(int ind, int my_left_bound, int my_right_bound)
        {
            int l = ind << 1;
            int r = l | 1;
            int mi = mid(my_left_bound, my_right_bound);
            int lson_length = (mi - my_left_bound + 1);
            int rson_length = (my_right_bound - mi);
            if (_D[ind].lazy_mul != 1)
            {
                _D[l].sum_content *= _D[ind].lazy_mul;
                _D[l].sum_content += _D[ind].lazy_add * lson_length;

                _D[r].sum_content *= _D[ind].lazy_mul;
                _D[r].sum_content += _D[ind].lazy_add * rson_length;

                _D[l].lazy_mul *= _D[ind].lazy_mul;
                _D[l].lazy_add *= _D[ind].lazy_mul;
                _D[l].lazy_add += _D[ind].lazy_add;

                _D[r].lazy_mul *= _D[ind].lazy_mul;
                _D[r].lazy_add *= _D[ind].lazy_mul;
                _D[r].lazy_add += _D[ind].lazy_add;

                _D[l].min_content *= _D[ind].lazy_mul;
                _D[l].min_content += _D[ind].lazy_add;

                _D[r].min_content *= _D[ind].lazy_mul;
                _D[r].min_content += _D[ind].lazy_add;

                _D[ind].lazy_mul = 1;
                _D[ind].lazy_add = 0;

                return;
            }
            if (_D[ind].lazy_add)
            {
                _D[l].sum_content += _D[ind].lazy_add * lson_length;
                _D[l].lazy_add += _D[ind].lazy_add;
                _D[r].sum_content += _D[ind].lazy_add * rson_length;
                _D[r].lazy_add += _D[ind].lazy_add;

                _D[l].min_content += _D[ind].lazy_add;
                _D[r].min_content += _D[ind].lazy_add;
                _D[ind].lazy_add = 0;
            }
        }

        void _query_sum(
            int l,
            int r,
            T &res,
            int node_l,
            int node_r,
            int x)
        {
            if (l <= node_l and node_r <= r)
            {
                res += _D[x].sum_content;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    _query_sum(l, r, res, node_l, mi, x << 1);
                if (r > mi)
                    _query_sum(l, r, res, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }
        void _query_min(
            int l,
            int r,
            T &res,
            int node_l,
            int node_r,
            int x)
        {
            if (l <= node_l and node_r <= r)
            {
                res = min(res, _D[x].min_content);
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    _query_min(l, r, res, node_l, mi, x << 1);
                if (r > mi)
                    _query_min(l, r, res, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        T query_sum(int l, int r)
        {
            T res = 0;
            _query_sum(l, r, res, 1, valid_len, 1);
            return res;
        }

        T query_min(int l, int r)
        {
            T res;
            memset(&res, 0x3f, sizeof(res));
            _query_min(l, r, res, 1, valid_len, 1);
            return res;
        }
    };
}
