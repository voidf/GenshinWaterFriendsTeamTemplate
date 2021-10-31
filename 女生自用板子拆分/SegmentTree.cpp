
//  2021.10.31 线段树支持使用矩阵
namespace Tree
{
#define Add0 0
#define Mul1 1

    // #define Add0 Geometry::Matrix<m998>(1, 3)
    // #define Mul1 Geometry::SquareMatrix<m998>::eye(3)
    template <typename T, typename Tadd = T, typename Tmul = T>
    struct _iNode
    {
        Tadd lazy_add;
        T sum_content;
        Tmul lazy_mul;
        // T max_content;
        T min_content;
        T sqrt_content;
        _iNode() : lazy_add(Add0), sum_content(Add0), lazy_mul(Mul1), min_content(Add0), sqrt_content(Add0) {}
    };

    template <typename T, typename Tadd = T, typename Tmul = T>
    struct SegmentTree
    {
        using _Node = _iNode<T, Tadd, Tmul>;
        int len;       // 线段树实际节点数
        int valid_len; // 原有效数据长度
        int QL, QR;    // 暂存询问避免递归下传
        Tmul MTMP;
		Tadd ATMP;
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

        void update_mul(int node_l, int node_r, int x)
        {
            if (QL <= node_l and node_r <= QR)
            {
                _D[x].lazy_add *= MTMP;
                _D[x].sum_content *= MTMP;
                _D[x].lazy_mul *= MTMP;
                _D[x].min_content *= MTMP;

                _D[x].sqrt_content = _D[x].sqrt_content * MTMP * MTMP;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (QL <= mi)
                    update_mul(node_l, mi, x << 1);
                if (QR > mi)
                    update_mul(mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void update_add(int node_l, int node_r, int x)
        {
            if (QL <= node_l and node_r <= QR)
            {
                int my_length = node_r - node_l + 1;
                _D[x].lazy_add += ATMP;

                _D[x].sqrt_content = _D[x].sqrt_content + 2 * ATMP * _D[x].sum_content + (ATMP * ATMP * my_length);

                _D[x].sum_content += ATMP * my_length;
                _D[x].min_content += ATMP;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (QL <= mi)
                    update_add(node_l, mi, x << 1);
                if (QR > mi)
                    update_add(mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void range_mul(int l, int r, const Tmul &v)
        {
            QL = l;
            QR = r;
            MTMP = v;
            update_mul(1, valid_len, 1);
        }

        void range_add(int l, int r, const Tadd &v)
        {
            QL = l;
            QR = r;
            ATMP = v;
            update_add(1, valid_len, 1);
        }

        inline void maintain(int i)
        {
            int l = i << 1;
            int r = l | 1;
            _D[i].sum_content = (_D[l].sum_content + _D[r].sum_content);
            _D[i].min_content = min(_D[l].min_content, _D[r].min_content);
            _D[i].sqrt_content = (_D[l].sqrt_content + _D[r].sqrt_content);
        }

        inline void push_down(int ind, int my_left_bound, int my_right_bound)
        {
            int l = ind << 1;
            int r = l | 1;
            int mi = mid(my_left_bound, my_right_bound);
            int lson_length = (mi - my_left_bound + 1);
            int rson_length = (my_right_bound - mi);
            if (_D[ind].lazy_mul != Mul1)
            {
                // 区间和
                _D[l].sum_content *= _D[ind].lazy_mul;

                _D[r].sum_content *= _D[ind].lazy_mul;

                _D[l].lazy_mul *= _D[ind].lazy_mul;
                _D[l].lazy_add *= _D[ind].lazy_mul;

                _D[r].lazy_mul *= _D[ind].lazy_mul;
                _D[r].lazy_add *= _D[ind].lazy_mul;

                // RMQ
                _D[l].min_content *= _D[ind].lazy_mul;

                _D[r].min_content *= _D[ind].lazy_mul;

                // 平方和，依赖区间和
                _D[l].sqrt_content = _D[l].sqrt_content * _D[ind].lazy_mul * _D[ind].lazy_mul;

                _D[r].sqrt_content = _D[r].sqrt_content * _D[ind].lazy_mul * _D[ind].lazy_mul;

                _D[ind].lazy_mul = Mul1;
            }
            if (_D[ind].lazy_add != Add0)
            {
                // 平方和，先于区间和处理
                _D[l].sqrt_content = _D[l].sqrt_content + 2 * _D[ind].lazy_add * _D[l].sum_content + _D[ind].lazy_add * _D[ind].lazy_add * lson_length;

                _D[r].sqrt_content = _D[r].sqrt_content + 2 * _D[ind].lazy_add * _D[r].sum_content + _D[ind].lazy_add * _D[ind].lazy_add * rson_length;

                _D[l].sum_content += _D[ind].lazy_add * lson_length;
                _D[l].lazy_add += _D[ind].lazy_add;
                _D[r].sum_content += _D[ind].lazy_add * rson_length;
                _D[r].lazy_add += _D[ind].lazy_add;

                _D[l].min_content += _D[ind].lazy_add;
                _D[r].min_content += _D[ind].lazy_add;
                _D[ind].lazy_add = Add0;
            }
        }

        void _query_sum(
            T &res,
            int node_l,
            int node_r,
            int x)
        {
            if (QL <= node_l and node_r <= QR)
            {
                res += _D[x].sum_content;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (QL <= mi)
                    _query_sum(res, node_l, mi, x << 1);
                if (QR > mi)
                    _query_sum(res, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }
        void _query_min(
            T &res,
            int node_l,
            int node_r,
            int x)
        {
            if (QL <= node_l and node_r <= QR)
            {
                res = min(res, _D[x].min_content);
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (QL <= mi)
                    _query_min(res, node_l, mi, x << 1);
                if (QR > mi)
                    _query_min(res, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        void _query_sqrt(
            T &res,
            int node_l,
            int node_r,
            int x)
        {
            if (QL <= node_l and node_r <= QR)
            {
                res += _D[x].sqrt_content;
            }
            else
            {
                push_down(x, node_l, node_r);
                int mi = mid(node_l, node_r);
                if (QL <= mi)
                    _query_sqrt(res, node_l, mi, x << 1);
                if (QR > mi)
                    _query_sqrt(res, mi + 1, node_r, x << 1 | 1);
                maintain(x);
            }
        }

        T query_sum(int l, int r)
        {
            T res = Add0;
            QL = l;
            QR = r;
            _query_sum(res, 1, valid_len, 1);
            return res;
        }

        T query_min(int l, int r)
        {
            T res;
            memset(&res, 0x3f, sizeof(res));
            QL = l;
            QR = r;
            _query_min(res, 1, valid_len, 1);
            return res;
        }

        T query_sqrt(int l, int r)
        {
            T res = Add0;
            QL = l;
            QR = r;
            _query_sqrt(res, 1, valid_len, 1);
            return res;
        }
    };
}
