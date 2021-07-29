
namespace Tree
{
    template <typename T>
    struct _iNode
    {
        T lazy_add;
        T sum_content;
        T lazy_mul;
#ifdef REQUIRE_RMQ
        T max_content;
        T min_content;
#endif
    };

    
    template <typename T>
    struct SegmentTree
    {
        using _Node = _iNode<T>;
        int len;        // 线段树实际节点数
        int valid_len;  // 原有效数据长度
        _Node *_start;  // 起始位置
        _Node *_finish; // 结束位置
        // template <typename AllocationPlaceType = void>
        SegmentTree(int length, void *arr = nullptr) // 构造函数只分配内存
        {
            valid_len = length;
            len = 1 << 1 + (int)ceil(log2(length));

            if (arr != nullptr)
            {
                _start = ::new (arr) _Node[len]; // 会占用arr序列的空间
            }
            else
            {
                _start = new _Node[len];
            }

            _finish = _start + len;
        }
        // ~SegmentTree() { delete[] _start; }

        _Node *begin() { return _start; }
        _Node *end() { return _finish; }

        void show()
        {
            std::cout << '[';
            for (_Node *i = begin(); i != end(); i++)
                std::cout << i->sum_content << ",]"[i == end() - 1] << " \n"[i == end() - 1];
        }

        static int mid(int l, int r) { return l + r >> 1; }

        void update_mul(int l,
                        int r,
                        T &&v,
                        int node_l,
                        int node_r,
                        int x)
        {
            if (l <= node_l and node_r <= r)
            {
                _start[x].lazy_add *= v;
                _start[x].sum_content *= v;
                _start[x].lazy_mul *= v;
                _start[x].min_content *= v;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    update_mul(l, r, std::move(v), node_l, mi, x << 1);
                if (r > mi)
                    update_mul(l, r, std::move(v), mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void update_add(int l,
                        int r,
                        T &&v,
                        int node_l,
                        int node_r,
                        int x)
        {
            if (l <= node_l and node_r <= r)
            {
                LL my_length = node_r - node_l + 1;
                _start[x].lazy_add += v;
                _start[x].sum_content += my_length * v;
                _start[x].min_content += v;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (l <= mi)
                    update_add(l, r, std::move(v), node_l, mi, x << 1);
                if (r > mi)
                    update_add(l, r, std::move(v), mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void range_mul(int l, int r, T &v)
        {
            update_mul(l, r, std::move(v), 1, valid_len, 1);
        }

        void range_mul(int l, int r, T &&v)
        {
            update_mul(l, r, std::move(v), 1, valid_len, 1);
        }

        void range_add(int l, int r, T &v)
        {
            update_add(l, r, std::move(v), 1, valid_len, 1);
        }

        void range_add(int l, int r, T &&v)
        {
            update_add(l, r, std::move(v), 1, valid_len, 1);
        }

        inline void maintain(int i)
        {
            int l = i << 1;
            int r = l | 1;
            _start[i].sum_content = (_start[l].sum_content + _start[r].sum_content);
            _start[i].min_content = min(_start[l].min_content, _start[r].min_content);
        }

        inline void push_down(int ind, int my_left_bound, int my_right_bound)
        {
            int l = ind << 1;
            int r = l | 1;
            int mi = mid(my_left_bound, my_right_bound);
            int lson_length = (mi - my_left_bound + 1);
            int rson_length = (my_right_bound - mi);
            if (_start[ind].lazy_mul != 1)
            {
                _start[l].sum_content *= _start[ind].lazy_mul;
                _start[l].sum_content += _start[ind].lazy_add * lson_length;

                _start[r].sum_content *= _start[ind].lazy_mul;
                _start[r].sum_content += _start[ind].lazy_add * rson_length;

                _start[l].lazy_mul *= _start[ind].lazy_mul;
                _start[l].lazy_add *= _start[ind].lazy_mul;
                _start[l].lazy_add += _start[ind].lazy_add;

                _start[r].lazy_mul *= _start[ind].lazy_mul;
                _start[r].lazy_add *= _start[ind].lazy_mul;
                _start[r].lazy_add += _start[ind].lazy_add;

                _start[l].min_content *= _start[ind].lazy_mul;
                _start[l].min_content += _start[ind].lazy_add;

                _start[r].min_content *= _start[ind].lazy_mul;
                _start[r].min_content += _start[ind].lazy_add;

                _start[ind].lazy_mul = 1;
                _start[ind].lazy_add = 0;

                return;
            }
            if (_start[ind].lazy_add)
            {
                _start[l].sum_content += _start[ind].lazy_add * lson_length;
                _start[l].lazy_add += _start[ind].lazy_add;
                _start[r].sum_content += _start[ind].lazy_add * rson_length;
                _start[r].lazy_add += _start[ind].lazy_add;

                _start[l].min_content += _start[ind].lazy_add;
                _start[r].min_content += _start[ind].lazy_add;
                _start[ind].lazy_add = 0;
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
                res += _start[x].sum_content;
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
                res = min(res, _start[x].min_content);
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


    // template <typename T>
    // struct type_deduce
    // {
    //     using typ = T;
    // };

//     template <typename T> // 使用的类T必须支持= （问就是你常用的int, long long, double甚至__int128都是支持的）
//     struct SegmentTree
//     {
//         using _Node = _iNode<T>;
//         int len;        // 线段树实际节点数
//         int valid_len;  // 原有效数据长度
//         _Node *_start;  // 起始位置
//         _Node *_finish; // 结束位置
//         // template <typename AllocationPlaceType = void>
//         SegmentTree(int length, void *arr = nullptr) // 构造函数只分配内存
//         {
//             valid_len = length;
//             len = 1 << 1 + (int)ceil(log2(length));

//             // while (length > 1)
//             // {
//             //     len += length;
//             //     length = length + 1 >> 1;
//             // }

//             if (arr != nullptr)
//             {
//                 _start = ::new (arr) _Node[len]; // 会占用arr序列的空间
//             }
//             else
//             {
//                 _start = new _Node[len];
//             }

//             _finish = _start + len;
//         }

//         _Node *begin() { return _start; }
//         _Node *end() { return _finish; }

//         static int mid(int l, int r) { return l + r >> 1; }

//         void show()
//         {
//             std::cout << '[';
//             for (_Node *i = begin(); i != end(); i++)
//                 std::cout << i->sum_content << ",]"[i == end() - 1] << " \n"[i == end() - 1];
//         }

//         std::function<void(int, T &&, int)> update_policies[2] =
//             {
//                 [&](int x, T &&v, int my_length)
//                 {
//                     _start[x].lazy_add *= v; // 更新此次修改的tag值
//                     _start[x].sum_content *= v;
//                     _start[x].lazy_mul *= v;
// #ifdef REQUIRE_RMQ
//                     _start[x].max_content *= v;
//                     _start[x].min_content *= v;
// #endif
// #ifdef MODULO
//                     _start[x].lazy_mul %= MODULO;
//                     _start[x].sum_content %= MODULO;
//                     _start[x].lazy_add %= MODULO;
// #endif
//                 },
//                 [&](int x, T &&v, int my_length)
//                 {
//                     _start[x].lazy_add += v; // 更新此次修改的tag值
//                     _start[x].sum_content += my_length * v;
// #ifdef REQUIRE_RMQ
//                     _start[x].max_content += v;
//                     _start[x].min_content += v;
// #endif
// #ifdef MODULO
//                     _start[x].sum_content %= MODULO;
//                     _start[x].lazy_add %= MODULO;
// #endif
//                 }};

//         std::function<void(int, T &)> query_policies[3] =
//             {
//                 [&](int x, T &res)
//                 {
//                     res += _start[x].sum_content;
// #ifdef MODULO
//                     res %= MODULO;
// #endif
//                 },
//                 [&](int x, T &res)
//                 {
// #ifdef REQUIRE_RMQ
//                     res = min(res, _start[x].min_content);
// #endif
//                 },
//                 [&](int x, T &res)
//                 {
// #ifdef REQUIRE_RMQ
//                     res = max(res, _start[x].max_content);
// #endif
//                 }};

//         template <typename Func>
//         void range_update(
//             int l,
//             int r,
//             T &&v,
//             int node_l,
//             int node_r,
//             int x,
//             Func &update_policy)
//         {
//             if (l <= node_l and node_r <= r)
//             {
//                 update_policy(x, std::move(v), node_r - node_l + 1);
//             }
//             else
//             {
//                 push_down(x, node_l, node_r);
//                 int mi = mid(node_l, node_r);
//                 if (l <= mi)
//                     range_update(l, r, std::move(v), node_l, mi, x << 1, update_policy);
//                 if (r > mi)
//                     range_update(l, r, std::move(v), mi + 1, node_r, x << 1 | 1, update_policy);
//                 maintain(x);
//             }
//         }

//         void range_mul(int l, int r, T &v)
//         {
//             range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[0]);
//         }

//         void range_mul(int l, int r, T &&v)
//         {
//             range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[0]);
//         }

//         void range_add(int l, int r, T &v)
//         {
//             range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[1]);
//         }

//         void range_add(int l, int r, T &&v)
//         {
//             range_update(l, r, std::move(v), 1, this->valid_len, 1, update_policies[1]);
//         }

//         inline void maintain(int i)
//         {
//             int l = i << 1;
//             int r = l | 1;
//             _start[i].sum_content = (_start[l].sum_content + _start[r].sum_content)
// #ifdef MODULO
//                                     % MODULO
// #endif
//                 ;
// #ifdef REQUIRE_RMQ
//             _start[i].max_content = max(_start[l].max_content, _start[r].max_content);
//             _start[i].min_content = min(_start[l].min_content, _start[r].min_content);
// #endif
//         }

//         void assign(T values[]) { build(values, 1, valid_len, 1); }

//         inline void build(T values[], int l, int r, int x)
//         {
//             _start[x].lazy_add = 0;
//             _start[x].lazy_mul = 1;
//             if (l == r)
//             {
//                 _start[x].sum_content = values[l - 1];
// #ifdef REQUIRE_RMQ
//                 _start[x].max_content = values[l - 1];
//                 _start[x].min_content = values[l - 1];
// #endif
//             }
//             else
//             {
//                 int mi = mid(l, r);
//                 build(values, l, mi, x << 1);
//                 build(values, mi + 1, r, x << 1 | 1);
//                 maintain(x);
//             }
//         }

//         inline void push_down(int ind, int my_left_bound, int my_right_bound)
//         {
//             int l = ind << 1;
//             int r = l | 1;
//             int mi = mid(my_left_bound, my_right_bound);
//             int lson_length = (mi - my_left_bound + 1);
//             int rson_length = (my_right_bound - mi);
//             if (_start[ind].lazy_mul != 1)
//             {
//                 _start[l].sum_content *= _start[ind].lazy_mul;
//                 _start[l].sum_content += _start[ind].lazy_add * lson_length;

//                 _start[r].sum_content *= _start[ind].lazy_mul;
//                 _start[r].sum_content += _start[ind].lazy_add * rson_length;

//                 _start[l].lazy_mul *= _start[ind].lazy_mul;
//                 _start[l].lazy_add *= _start[ind].lazy_mul;
//                 _start[l].lazy_add += _start[ind].lazy_add;

//                 _start[r].lazy_mul *= _start[ind].lazy_mul;
//                 _start[r].lazy_add *= _start[ind].lazy_mul;
//                 _start[r].lazy_add += _start[ind].lazy_add;
// #ifdef MODULO
//                 _start[l].lazy_mul %= MODULO;
//                 _start[l].lazy_add %= MODULO;
//                 _start[l].sum_content %= MODULO;

//                 _start[r].lazy_mul %= MODULO;
//                 _start[r].lazy_add %= MODULO;
//                 _start[r].sum_content %= MODULO;
// #endif

// #ifdef REQUIRE_RMQ
//                 _start[l].max_content *= _start[ind].lazy_mul;
//                 _start[l].max_content += _start[ind].lazy_add;
//                 _start[l].min_content *= _start[ind].lazy_mul;
//                 _start[l].min_content += _start[ind].lazy_add;

//                 _start[r].max_content *= _start[ind].lazy_mul;
//                 _start[r].max_content += _start[ind].lazy_add;
//                 _start[r].min_content *= _start[ind].lazy_mul;
//                 _start[r].min_content += _start[ind].lazy_add;
// #endif
//                 _start[ind].lazy_mul = 1;
//                 _start[ind].lazy_add = 0;

//                 return;
//             }
//             if (_start[ind].lazy_add)
//             {
//                 _start[l].sum_content += _start[ind].lazy_add * lson_length;
//                 _start[l].lazy_add += _start[ind].lazy_add;
//                 _start[r].sum_content += _start[ind].lazy_add * rson_length;
//                 _start[r].lazy_add += _start[ind].lazy_add;
// #ifdef MODULO
//                 _start[l].lazy_add %= MODULO;
//                 _start[l].sum_content %= MODULO;
//                 _start[r].lazy_add %= MODULO;
//                 _start[r].sum_content %= MODULO;
// #endif

// #ifdef REQUIRE_RMQ
//                 _start[l].max_content += _start[ind].lazy_add;
//                 _start[l].min_content += _start[ind].lazy_add;

//                 _start[r].max_content += _start[ind].lazy_add;
//                 _start[r].min_content += _start[ind].lazy_add;
// #endif
//                 _start[ind].lazy_add = 0;
//             }
//         }

//         template <typename Func>
//         void query_proxy(
//             int l,
//             int r,
//             T &res,
//             int node_l,
//             int node_r,
//             int x,
//             Func &query_policy)
//         {
//             if (l <= node_l and node_r <= r)
//             {
//                 query_policy(x, res);
//             }
//             else
//             {
//                 push_down(x, node_l, node_r);
//                 int mi = mid(node_l, node_r);
//                 if (l <= mi)
//                     query_proxy(l, r, res, node_l, mi, x << 1, query_policy);
//                 if (r > mi)
//                     query_proxy(l, r, res, mi + 1, node_r, x << 1 | 1, query_policy);
//                 maintain(x);
//             }
//         }

//         T query_sum(int l, int r)
//         {
//             T res = 0;
//             query_proxy(l, r, res, 1, valid_len, 1, query_policies[0]);
//             return res;
//         }
// #ifdef REQUIRE_RMQ
//         T query_max(int l, int r)
//         {
//             T res = 0;
//             query_proxy(l, r, res, 1, valid_len, 1, query_policies[2]);
//             return res;
//         }

//         T query_min(int l, int r)
//         {
//             T res;
//             memset(&res, 0x3f, sizeof(res));
//             query_proxy(l, r, res, 1, valid_len, 1, query_policies[1]);
//             return res;
//         }
// #endif
//     };

} // namespace Tree
