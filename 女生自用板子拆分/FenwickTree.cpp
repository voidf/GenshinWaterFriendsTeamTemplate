/**  @author:        voidf 
*    @tested_date:   2021.1.26
*    树状数组两个板子：RMQ版和求和版
*
*    (过)(度)(封)(装)
*     面  向  对  象 设计，需要C++11
*
*    *本板子中虽然没有用private限制，但所有带inline的方法都是不推荐外部调用的
*
*    求和版api:
*        创建一个树状数组:
*            FenwickTree::FenwickTree_Sum<类型> 树状数组的名字(源序列, 长度, 是否在源序列上构建线段树)
*        用例1：
*            long long *arr = new long long[n << 5];
*            FenwickTree::FenwickTree_Sum<long long> tree(arr, n, true);
*        用例2：
*            FenwickTree::FenwickTree_Sum<long long> tree(arr, n, false);
*
*            # 构造函数最后一个参数为true时，会根据提供的序列直接在序列地址上建树（区别于ZKW线段树的板子，这里并不是直接读入树状数组）
*            # 其中T是你指定的类型，可以是long long, int, double, float这些基本类型，如果要传入类请根据底下说明重载一些运算符
*            
*            # 如果为false，则线段树会自动申请一块足够的空间，并用源序列中的数据建树
*            # 初始化时所传源序列下标从0开始，但实例化的树中下标都是以1开始
*            # 注意这样不能空间复用，多组样例情况下慎用
*        单点加法：
*            tree.add(position, value)
*            # 向树中下标position处元素加上value值，下标从1开始
*        单点修改：
*            tree.modify(position, value)
*            # 向树中下标position处元素修改为value值，下标从1开始
*        求前缀和：
*            tree.get_prefix(ind)
*            # 求树中下标position（包含）处往前所有元素的和，下标从1开始
*        区间求和：
*            tree.query(l, r)
*            # 求树中下标闭区间[l, r]中所有元素的和，下标从1开始
*        打印树状数组：
*            tree.show()
*        遍历树状数组：
*            for(auto i:tree)
*
*    RMQ版api:
*        创建一个树状数组:
*            FenwickTree::FenwickTree_Max<类型> 树状数组的名字(源序列, 长度, 是否在源序列上构建线段树)
*            # **RMQ型树状数组需要指定源序列以保证正确性**
*        指定源序列：
*            set_origin(arr)
*        区间最大值：
*            tree.query(l, r)
*            # 求树中闭区间[l,r]中所有元素的最大值，下标从1开始
*        （如果需要区间最小值，将本板子中所有std::max换成std::min即可）
*        单点加法：
*            tree.add(position, value)
*            # 向树中下标position处元素加上value值，下标从1开始
*        单点修改：
*            tree.modify(position, value)
*            # 向树中下标position处元素修改为value值，下标从1开始
*        打印树状数组：
*            tree.show()
*        遍历树状数组：
*            for(auto i:tree)
*
*    实战用例可以参考main函数中的几题
**/
#include <iostream>
#include <vector>
#include <cstring>
namespace FenwickTree // 树状数组
{
    template <typename T> // 使用的类T必须支持= （问就是你常用的int, long long, double甚至__int128都是支持的）
    struct FenwickTree_Base
    {
        int len = 0; // 数组大小
        T *_start;   // 起始位置
        T *_finish;  // 结束位置
        virtual void build() = 0;
        // virtual void modify(int ind, T &x) = 0;
        // virtual T query(int l, int r) = 0;                               // 左闭右闭的区间查询，下标从1开始
        FenwickTree_Base(T arr[], int length, bool direct_build = false) // 没法直接通过数组获得长度，得另外传
        //direct_build为true时直接在原地址上建树状数组，避免new消耗时间
        {
            len = length;
            if (direct_build)
                _start = arr;
            else
            {
                _start = new T[len];
                for (int idx = 0; idx < len; idx++)
                    _start[idx] = arr[idx];
            }
            _finish = _start + len;
        }
        FenwickTree_Base(T *begin, T *end, bool direct_build = false) // 起始指针和结束指针（左闭右开）
        {
            len = end - begin;
            if (direct_build)
                _start = begin;
            else
            {
                _start = new T[len];
                int idx = 0;
                for (auto i = begin; i != end; i++)
                    _start[idx++] = *i;
            }
            _finish = end;
        }
        FenwickTree_Base(std::vector<T> &v) // 对std::vector的支持，其他容器请自己写（
        {
            len = v.size();
            _start = new T[len];
            _finish = _start + len;
            int idx = 0;
            for (auto &i : v)
                _start[idx++] = i;
        }
        T *begin() { return _start; }
        T *end() { return _finish; } // 这两句用来支持foreach写法
        void show()                  // 打印树状数组内容
        {
            std::cout << '[';
            for (auto i = 0; i < len; i++)
                std::cout << _start[i] << ",]"[i == len - 1] << " \n"[i == len - 1];
        }
        inline int lowbit(int x) { return x & -x; } // 不会有人开1e9以上的数组吧？不会吧不会吧？
    };
    template <typename T>                               // 类型T必须支持+= -
    struct FenwickTree_Sum : public FenwickTree_Base<T> // 求和型树状数组
    {
        typedef FenwickTree_Base<T> _Base;
        using _Base::_start;
        using _Base::len;
        using _Base::lowbit;
        template <typename... Args>
        FenwickTree_Sum(Args &&...args) : _Base(std::forward<Args>(args)...) { build(); } // 把当前构造函数的参数扔给父构造函数做，然后建树
        inline void build()
        {
            for (int i = len; i > 0; i--) // 倒序建树防止重复计算
                for (int j = i + lowbit(i); j <= len; j += lowbit(j))
                    _start[j - 1] += _start[i - 1];
        }
        void modify(int ind, T &x) // 在ind下标处将值修改为x，下标(ind)从1开始
        {
            modify(ind, std::move(x));
        }
        void modify(int ind, T &&x) // 为了兼容右值引用，所以复制了一份
        {
            T diff = x - _start[ind - 1];
            while (ind <= len)
            {
                _start[ind - 1] += diff;
                ind += lowbit(ind);
            }
        }
        void add(int ind, T &x) { modify(ind, _start[ind - 1] + x); } // 在ind下标处将值+=x，下标(ind)从1开始
        void add(int ind, T &&x) { modify(ind, _start[ind - 1] + x); }
        T get_prefix(int pos)
        {
            T x = _start[pos - 1];
            while (pos > 0)
                x += _start[(pos -= lowbit(pos)) - 1];
            return x;
        }
        T query(int l, int r) { return get_prefix(r) - get_prefix(l - 1); }
    };

    template <typename T>                               // 类型T必须支持<运算符
    struct FenwickTree_Max : public FenwickTree_Base<T> // 求最大值型树状数组，最小值直接把本板子底下所有std::max换成std::min即可
    {
        typedef FenwickTree_Base<T> _Base;
        using _Base::_start;
        using _Base::len;
        using _Base::lowbit;
        T *_origin;
        template <typename... Args>
        FenwickTree_Max(Args &&...args) : _Base(std::forward<Args>(args)...) { build(); }

        void set_origin(T *p) { _origin = p; } // 设置原数组

        inline void build()
        {
            for (int i = len; i > 0; i--)
                for (int j = i + lowbit(i); j <= len; j += lowbit(j))
                    _start[j - 1] = std::max(_start[j - 1], _start[i - 1]);
        }
        void modify(int ind, T &x, T *origin = NULL) // 将ind位置修改为x，下标还是从1开始，因为查询用到，所以要传入原序列origin一起改
        {
            modify(ind, std::move(x), origin);
        }
        void modify(int ind, T &&x, T *origin = NULL)
        {
            if (not origin)
                origin = _origin;
            origin[ind - 1] = x;
            _start[ind - 1] = x;
            while (ind <= len)
            {
                int lowest = lowbit(ind);
                for (auto i = 1; i < lowest; i <<= 1)
                    _start[ind - 1] = std::max(_start[ind - i - 1], _start[ind - 1]);
                ind += lowbit(ind);
            }
        }
        T get_prefix(int pos)
        {
            T x = _start[pos - 1];
            while (pos > 0)
                x = std::max(x, _start[(pos -= lowbit(pos)) - 1]);
            return x;
        }
        T query(int l, int r, T *origin = NULL) // 查询[l,r]内的最大值，需要传入原序列
        {
            if (not origin)
                origin = _origin;
            T res = origin[r - 1];
            while (l < r)
            {
                for (--r; r - l >= lowbit(r); r -= lowbit(r)) // 若能跳一个区间，则跳一个区间
                    res = std::max(res, _start[r - 1]);
                res = std::max(res, origin[r - 1]); // 否则跳一个点
            }
            return res;
        }
    };
} // namespace FenwickTree

int arr[200010]; // 静态分配
int tree[200010];
signed main()
{
    long long arrr[] = {1, 1, 4, 5, 1, 4, 19, 19, 810};
    FenwickTree::FenwickTree_Sum<long long> f(arrr, 10);
    f.show();
    for (auto i : f)
        std::cout << i; // 这是一般用例演示

#ifdef P3374 // 洛谷P3374的用例，原提交https://www.luogu.com.cn/record/45467470
    int n, m;
    std::ios::sync_with_stdio(false); // 能优化掉0.7s(37.4%)
    std::cin >> n >> m;
    long long *arr = new long long[n]; // 比较优雅的动态分配，但在P3374的表现中比上面的静态开数组慢0.06s
    for (auto i = 0; i < n; i++)
        std::cin >> arr[i];
    FenwickTree::FenwickTree_Sum<long long> f(arr, n, true); // 指定为true会影响原数组，比不指定快0.17s
    while (m--)
    {
        int arg1;
        int arg2;
        long long arg3;
        std::cin >> arg1 >> arg2 >> arg3;
        if (arg1 == 1)
            f.add(arg2, arg3);
        else
            std::cout << f.query(arg2, arg3) << '\n';
    }
#endif

#ifdef P3368 // https://www.luogu.com.cn/record/45468935                                                                                                                                             \
             // 如果把数组看成一个连续函数，那么差分的思想其实和求导差不多                                                                                              \
             // 对一个数组求差分以后如果需要对其某一段区间加上一个值，那么可以在其差分区间开头点加上这个值，在结尾点（不包含）减去这个值 \
             // 如果要求出原数的话需要“积分”，即差分数组的前缀和就是这个位置的原数

    int n, m;
    std::ios::sync_with_stdio(false);
    std::cin >> n >> m;
    long long *arr = new long long[n];
    for (auto i = 0; i < n; i++)
        std::cin >> arr[i];
    for (int i = n - 1; i >= 0; i--) // 倒序遍历的时候不建议用auto，类型推导如果推成无符号数可能形成死循环
        arr[i] -= arr[i - 1];
    FenwickTree::FenwickTree_Sum<long long> f(arr, n, true);
    while (m--)
    {
        int arg1;
        int arg2;
        int arg3;
        long long arg4;
        std::cin >> arg1;
        if (arg1 == 1)
        {
            std::cin >> arg2 >> arg3 >> arg4;
            f.add(arg2, arg4);
            f.add(arg3 + 1, -arg4);
        }
        else
        {
            std::cin >> arg2;
            std::cout << f.get_prefix(arg2) << '\n';
        }
    }
#endif
#ifdef HDU1754 // cin 686ms
    int n, m;
    std::ios::sync_with_stdio(false);
    while (std::cin >> n >> m)
    {
        for (auto i = 0; i < n; i++)
        {
            std::cin >> arr[i];
            tree[i] = arr[i];
        }
        FenwickTree::FenwickTree_Max<int> f(tree, n, true);
        f.set_origin(arr);
        while (m--)
        {
            char cmd;
            int a1, a2;
            std::cin >> cmd >> a1 >> a2;
            if (cmd == 'Q')
                std::cout << f.query(a1, a2) << '\n';
            else
                f.modify(a1, a2);
        }
    }
#endif
    return 0;
}