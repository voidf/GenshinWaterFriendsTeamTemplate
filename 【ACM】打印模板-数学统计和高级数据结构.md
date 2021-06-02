# 数据结构

## 并查集

```cpp
void init(int n)
{
    for(int i=0;i<n;i++)
    {
        parent[i]=i;
        rank[i]=0;   // 初始树的高度为0
    }
}
// 合并x和y所属的集合
void unite(int x,int y)
{
    x=find(x);
    y=find(y);
    if(x==y) return ;
    if(rank[x]<rank[y])
        parent[x]=y;  // 合并是从rank小的向rank大的连边
    else
    {
        parent[y]=x;
        if(rank[x]==rank[y]) rank[x]++;
    }
}
int find(int x)       //查找x元素所在的集合,回溯时压缩路径
{
    if (x != parent[x])
    {
        parent[x] = find(parent[x]);     //回溯时的压缩路径
    }         //从x结点搜索到祖先结点所经过的结点都指向该祖先结点
    return parent[x];
}
```

## 树状数组

```cpp
int n;
int a[1005],c[1005]; //对应原数组和树状数组

int lowbit(int x){
    return x&(-x);
}

void updata(int i,int k){    //在i位置加上k
    while(i <= n){
        c[i] += k;
        i += lowbit(i);
    }
}

int getsum(int i){        //求A[1 - i]的和
    int res = 0;
    while(i > 0){
        res += c[i];
        i -= lowbit(i);
    }
    return res;
}
```

## 树状数组（区间查询区间修改）

```c++
#include<iostream>
#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;
const int maxn=1e5+10;
int sum1[maxn];
int sum2[maxn];
int a[maxn];
int n,m;
int lowbit(int x){
  return x&(-x);
}
void update(int x,int w){//更新效果：把x位置后面所有的数的值+w
   for (int i=x;i<=n;i+=lowbit(i)){
      sum1[i]+=w;//维护前缀和c[i]
      sum2[i]+=w*(x-1);//维护前缀和c[i]*(n-1)
   }
}
void range_update(int l,int r,int val)//更新效果：把l位置到r位置所有的数的值+w
{
    update(l,val);
    update(r+1,-val);
}
int sum(int x){//求1-x的和
  int ans=0;
  for (int i=x;i>0;i-=lowbit(i)){
    ans+=x*sum1[i]-sum2[i];
  }
  return ans;
}
int range_ask(int l,int r){//求l-r的和
   return sum(r)-sum(l-1);
}
int main(){
  while(~scanf("%d%d",&n,&m)){
    for (int i=1;i<=n;i++){
        scanf("%d",&a[i]);
        update(i,a[i]-a[i-1]);//维护差分数组
    }
   }
  return 0;
}
```



## 线段树//待替换

```cpp
struct segtreeModel
{
    LL val;
    LL lazy_tag;//Tag一定要本节点做完再打，否则后面的更新不了前面的
    LL mul_tag;
    LL l_range, r_range;
    segtreeModel *l, *r;
} T[M << 2];

void build(const LL &rg1, const LL &rg2, const LL ind) //从0开始的线段树()
{
    LL mid = (rg1 + rg2) >> 1;
    T[ind].lazy_tag = 0;
    T[ind].mul_tag = 1;
    T[ind].l_range = rg1;
    T[ind].r_range = rg2;

    if (rg1 == rg2)
    {
        T[ind].val = Array[rg1];
        return;
    }
    T[ind].l = &T[(ind << 1) + 1];
    T[ind].r = &T[(ind << 1) + 2];
    build(rg1, mid, (ind << 1) + 1);
    build(mid + 1, rg2, (ind << 1) + 2);

    T[ind].val = DSUM(T[ind].l->val, T[ind].r->val);
    DMO(T[ind].val, mo);
}

void push_down(segtreeModel *ind)
{
    segtreeModel *lson = ind->l;
    segtreeModel *rson = ind->r;
    lson->val = (lson->val * ind->mul_tag + ind->lazy_tag * (lson->r_range + 1 - lson->l_range)) % mo;
    rson->val = (rson->val * ind->mul_tag + ind->lazy_tag * (rson->r_range + 1 - rson->l_range)) % mo;

    lson->mul_tag *= ind->mul_tag;
    DMO(lson->mul_tag, mo);
    rson->mul_tag *= ind->mul_tag;
    DMO(rson->mul_tag, mo);

    lson->lazy_tag = (ind->lazy_tag + lson->lazy_tag * ind->mul_tag) % mo;
    rson->lazy_tag = (ind->lazy_tag + rson->lazy_tag * ind->mul_tag) % mo;

    ind->lazy_tag = 0;
    ind->mul_tag = 1;
}

void modify(const LL &rg1, const LL &rg2, segtreeModel *ind, const LL v)
{

    if (rg1 <= ind->l_range && ind->r_range <= rg2) //魔改区间内
    {

        if (mulFlg)
        {

            ind->mul_tag *= v;
            ind->val *= v;
            ind->lazy_tag *= v;
            DMO(ind->lazy_tag, mo);
            DMO(ind->val, mo);
            DMO(ind->mul_tag, mo);
        }
        else
        {
            ind->val += (v * (ind->r_range - ind->l_range + 1));
            ind->lazy_tag += v;
            DMO(ind->lazy_tag, mo);
            DMO(ind->val, mo);
        }
    }
    else //部分在魔改区间内
    {
        push_down(ind);
        //LL mid = (ind->l_range + ind->r_range) >> 1;
        if (ind->l->r_range >= rg1)
            modify(rg1, rg2, ind->l, v);
        if (ind->r->l_range <= rg2)
            modify(rg1, rg2, ind->r, v);
        ind->val = DSUM(ind->l->val, ind->r->val) % mo;
    }
}

LL query(const LL &rg1, const LL &rg2, segtreeModel *ind)
{
    // if (ind->l_range > rg2 || ind->r_range < rg1)
    //     return 0;
    if (rg1 <= ind->l_range && ind->r_range <= rg2)
    {
        return ind->val;
    }
    else
    {
        push_down(ind);
        //LL mid = (ind->l_range + ind->r_range) >> 1;
        LL v1 = ind->l->r_range >= rg1 ? query(rg1, rg2, ind->l) : 0;
        LL v2 = ind->r->l_range <= rg2 ? query(rg1, rg2, ind->r) : 0;
        return DSUM(v1, v2) % mo;
    }
}
```

## 主席树（非工程版）

```c++
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;
const int maxn = 1e5;  // 数据范围
int tot, n, m;
int sum[(maxn << 5) + 10], rt[maxn + 10], ls[(maxn << 5) + 10],
    rs[(maxn << 5) + 10];
int a[maxn + 10], ind[maxn + 10], len;
inline int getid(const int &val) {  // 离散化
  return lower_bound(ind + 1, ind + len + 1, val) - ind;
}
int build(int l, int r) {  // 建树
  int root = ++tot;
  if (l == r) return root;
  int mid = l + r >> 1;
  ls[root] = build(l, mid);
  rs[root] = build(mid + 1, r);
  return root;  // 返回该子树的根节点
}
int update(int k, int l, int r, int root) {  // 插入操作
  int dir = ++tot;
  ls[dir] = ls[root], rs[dir] = rs[root], sum[dir] = sum[root] + 1;
  if (l == r) return dir;
  int mid = l + r >> 1;
  if (k <= mid)
    ls[dir] = update(k, l, mid, ls[dir]);
  else
    rs[dir] = update(k, mid + 1, r, rs[dir]);
  return dir;
}
int query(int u, int v, int l, int r, int k) {  // 查询操作
  int mid = l + r >> 1,
      x = sum[ls[v]] - sum[ls[u]];  // 通过区间减法得到左儿子的信息
  if (l == r) return l;
  if (k <= x)  // 说明在左儿子中
    return query(ls[u], ls[v], l, mid, k);
  else  // 说明在右儿子中
    return query(rs[u], rs[v], mid + 1, r, k - x);
}
inline void init() {
  scanf("%d%d", &n, &m);
  for (int i = 1; i <= n; ++i) scanf("%d", a + i);
  memcpy(ind, a, sizeof ind);
  sort(ind + 1, ind + n + 1);
  len = unique(ind + 1, ind + n + 1) - ind - 1;
  rt[0] = build(1, len);
  for (int i = 1; i <= n; ++i) rt[i] = update(getid(a[i]), 1, len, rt[i - 1]);
}
int l, r, k;
inline void work() {
  while (m--) {
    scanf("%d%d%d", &l, &r, &k);
    printf("%d\n", ind[query(rt[l - 1], rt[r], 1, len, k)]);  // 回答询问
  }
}
int main() {
  init();
  work();
  return 0;
}
```





## 主席树

```cpp
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
        int l, r;
        Node() : meta(), l(-1), r(-1) {}
        void at_build()
        {
            // memset(meta, 0, sizeof(meta));
            meta = 0;
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

    int legacy_query(int l_bound, int r_bound, int ql, int qr, int cur)
    {
        Node &me = nodes[cur];
        if (l_bound >= ql and r_bound <= qr)
            return me.meta;
        int res = 0;
        int m = mid(l_bound, r_bound);
        if (ql <= m)
            res += legacy_query(l_bound, m, ql, qr, me.l);
        if (qr >= m + 1)
            res += legacy_query(m + 1, r_bound, ql, qr, me.r);
        return res;
    }
};
```

## 树链剖分//？

```cpp
LL ARRAY[M << 1];
LL CNT = 0;
LL n;
LL mod;

struct NodeModel
{
    LL size, height, dfs_order, value;
    NodeModel *son, *fa, *top;
    vector<NodeModel *> E;
    void d1(NodeModel *father)
    {
        this->size = 1;
        this->son = NULL;
        this->fa = father;
        this->height = father->height + 1;
        for (auto cur : this->E)
        {
            // auto dst = N[E[cur].to];
            if (cur != father)
            {
                cur->d1(this);
                this->size += cur->size;
                if (this->son == NULL || cur->size > this->son->size)
                    this->son = cur;
            }
        }
    }
    void d2(NodeModel *father, NodeModel *k)
    {
        this->top = k;
        this->dfs_order = CNT;
        ARRAY[CNT++] = this->value;
        if (this->son != NULL)
            this->son->d2(this, k);
        for (auto i : this->E)
            if (i != father && i != this->son)
                i->d2(this, i);
    }

    void add_edge(NodeModel *dst)
    {
        this->E.push_back(dst);
    }
} N[M << 2];

LL NODECNT = 0;
struct SegmentTreeNode
{
    SegmentTreeNode *l, *r;
    LL value, lazy, max_value;
    LL query_sum(LL rg1, LL rg2, LL operation_l, LL operation_r, LL add_val)
    {
        if (operation_l > operation_r)
            swap(operation_l, operation_r);
        if (operation_l <= rg1 && rg2 <= operation_r)
            return (this->value + (LL)(rg2 - rg1 + 1) * (add_val % mod)) % mod;
        else
        {
            LL v1 = (DMID(rg1, rg2) >= operation_l) ? this->l->query_sum(rg1, DMID(rg1, rg2), operation_l, operation_r, (add_val + this->lazy) % mod) : 0;
            LL v2 = (DMID(rg1, rg2) + 1 <= operation_r) ? this->r->query_sum(DMID(rg1, rg2) + 1, rg2, operation_l, operation_r, (add_val + this->lazy) % mod) : 0;
            return ((v1 + v2) % mod + mod) % mod;
        }
    }
    LL query_max(LL rg1, LL rg2, LL operation_l, LL operation_r, LL add_val)
    {
        if (operation_l > operation_r)
            swap(operation_l, operation_r);
        if (operation_l <= rg1 && rg2 <= operation_r)
            return this->max_value /*+ add_val*/;
        else
        {
            LL v1 = (DMID(rg1, rg2) >= operation_l) ? this->l->query_max(rg1, DMID(rg1, rg2), operation_l, operation_r, add_val + this->lazy) : -INF;
            LL v2 = (DMID(rg1, rg2) + 1 <= operation_r) ? this->r->query_max(DMID(rg1, rg2) + 1, rg2, operation_l, operation_r, add_val + this->lazy) : -INF;
            return max(v1, v2);
        }
    }
    void modify(LL rg1, LL rg2, LL operation_l, LL operation_r, LL x) // 单点修改
    {
        if (operation_l <= rg1 && rg2 <= operation_r)
        {
            this->value += x * (rg2 - rg1 + 1);
            this->value %= mod;
            this->lazy += x;
            this->lazy %= mod;
            return;
        }
        if (operation_l <= DMID(rg1, rg2))
            this->l->modify(rg1, DMID(rg1, rg2), operation_l, operation_r, x);
        if (DMID(rg1, rg2) + 1 <= operation_r)
            this->r->modify(DMID(rg1, rg2) + 1, rg2, operation_l, operation_r, x);
        this->value = this->l->value + this->r->value + this->lazy * (rg2 - rg1 + 1);
        this->value %= mod;
        // this->max_value = max(this->l->max_value, this->r->max_value);
    }

} T[M];
SegmentTreeNode *build(LL rg1, LL rg2)
{
    LL tmp = NODECNT++;
    if (rg2 > rg1)
    {
        T[tmp].l = build(rg1, DMID(rg1, rg2));
        T[tmp].r = build(DMID(rg1, rg2) + 1, rg2);
        T[tmp].value = (T[tmp].l->value + T[tmp].r->value) % mod;
        T[tmp].max_value = max(T[tmp].l->max_value, T[tmp].r->max_value);
    }
    else
    {
        T[tmp].value = T[tmp].max_value = ARRAY[rg1];
    }
    T[tmp].lazy = 0;
    return &T[tmp];
}

LL query_SUM(NodeModel *u, NodeModel *v)
{
    LL ans = 0;
    while (u->top != v->top) // top相同应该是在一条链上
    {
        if (u->top->height < v->top->height)
            swap(u, v);
        ans += T[0].query_sum(0, n - 1, u->top->dfs_order, u->dfs_order, 0);
        ans %= mod;
        u = u->top->fa;
    }
    if (v->height > u->height)
        swap(u, v);
    ans += T[0].query_sum(0, n - 1, v->dfs_order, u->dfs_order, 0);
    ans %= mod;
    return (ans + mod) % mod;
}

void MODIFY(NodeModel *u, NodeModel *v, LL x)
{
    LL ans = 0;
    while (u->top != v->top) // top相同应该是在一条链上
    {
        if (u->top->height < v->top->height)
            swap(u, v);
        T[0].modify(0, n - 1, u->top->dfs_order, u->dfs_order, x);
        u = u->top->fa;
    }
    if (v->height > u->height)
        swap(u, v);
    T[0].modify(0, n - 1, v->dfs_order, u->dfs_order, x);
}

LL query_MAX(NodeModel *u, NodeModel *v)
{
    LL ans = -INF;
    while (u->top != v->top) // top相同应该是在一条链上
    {
        if (u->top->height < v->top->height)
            swap(u, v);
        ans = max(ans, T[0].query_max(0, n - 1, u->top->dfs_order, u->dfs_order, 0));
        u = u->top->fa;
    }
    if (v->height > u->height)
        swap(u, v);
    ans = max(ans, T[0].query_max(0, n - 1, v->dfs_order, u->dfs_order, 0));
    return ans;
}
char cmd[10];
signed main()
{
    n = qr();
    LL m = qr();
    LL root = qr();
    mod = qr();
    CNT = NODECNT = 0;
    mem(T, 0);
    mem(ARRAY, 0);
    mem(N, 0);

    for (auto i = 1; i <= n; i++)
    {
        N[i].value = qr();
    }
    for (auto i = 1; i < n; i++)
    {
        LL src = qr();
        LL dst = qr();
        N[src].add_edge(&N[dst]);
        N[dst].add_edge(&N[src]);
    }
    N[root].d1(&N[0]);
    N[root].d2(&N[0], &N[root]);
    build(0, n - 1);
    LL o1, o2, o3, o4;
    while (m--)
    {
        o1 = qr();
        switch (o1)
        {
        case 1:
            o2 = qr();
            o3 = qr();
            o4 = qr();
            MODIFY(&N[o2], &N[o3], o4);
            break;
        case 2:
            o2 = qr();
            o3 = qr();
            printf("%lld\n", query_SUM(&N[o2], &N[o3]) % mod);
            break;
        case 3:
            o2 = qr();
            o3 = qr();
            T[0].modify(0, n - 1, N[o2].dfs_order, N[o2].dfs_order + N[o2].size - 1, o3);
            break;
        default:
            o2 = qr();
            printf("%lld\n", T[0].query_sum(0, n - 1, N[o2].dfs_order, N[o2].dfs_order + N[o2].size - 1, 0) % mod);
            break;
        }
    }
    return 0;
}
```

## Splay

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 100005;
int rt, tot, fa[N], ch[N][2], val[N], cnt[N], sz[N],mark[N];
int n, m;
struct Splay
{
    void maintain(int x) { sz[x] = sz[ch[x][0]] + sz[ch[x][1]] + cnt[x]; }
    bool get(int x) { return x == ch[fa[x]][1]; }
    void pushdown(int x)
    {
        if(mark[x])
        {
            mark[ch[x][0]]^=1;
            mark[ch[x][1]]^=1;
            mark[x]=0;
            swap(ch[x][1],ch[x][0]);
        }
    }
    void clear(int x)
    {
        ch[x][0] = ch[x][1] = fa[x] = val[x] = sz[x] = cnt[x] = 0;
    }
    void write(int u)
    {
        pushdown(u);
        if(ch[u][0])
            write(ch[u][0]);
        if(val[u]>1&&val[u]<n+2)
            cout<<val[u]-1<<" ";

        if(ch[u][1])
            write(ch[u][1]);

    }
    void rotate(int x)
    {
        int y = fa[x], z = fa[y], chk = get(x);
        ch[y][chk] = ch[x][chk ^ 1];
        if (ch[x][chk ^ 1])
            fa[ch[x][chk ^ 1]] = y;
        ch[x][chk ^ 1] = y;
        fa[y] = x;
        fa[x] = z;
        if (z)
            ch[z][y == ch[z][1]] = x;
        maintain(x);
        maintain(y);
    }
    inline void splay(int x,int goal)
    {
        while(fa[x]!=goal)
        {
            int y=fa[x];int z=fa[y];
            if(z!=goal)
                (ch[z][1]==y)^(ch[y][1]==x)?rotate(x):rotate(y);
            rotate(x);
        }
        if(goal==0)
            rt=x;
    }
    void ins(int k)
    {
        if (!rt)
        {
            val[++tot] = k;
            cnt[tot]++;
            rt = tot;
            maintain(rt);
            return;
        }
        int cur = rt, f = 0;
        while (1)
        {
            if (val[cur] == k)
            {
                cnt[cur]++;
                maintain(cur);
                maintain(f);
                splay(cur,0);
                break;
            }
            f = cur;
            cur = ch[cur][val[cur] < k];
            if (!cur)
            {
                val[++tot] = k;
                cnt[tot]++;
                fa[tot] = f;
                ch[f][val[f] < k] = tot;
                maintain(tot);
                maintain(f);
                splay(tot,0);
                break;
            }
        }
    }
    int kth(int k)
    {
        int cur = rt;
        while (1)
        {
            pushdown(cur);
            if (ch[cur][0] && k <= sz[ch[cur][0]])
            {
                cur = ch[cur][0];
            }
            else
            {
                k -= cnt[cur] + sz[ch[cur][0]];
                if (k <= 0)
                {
                    //splay(cur,0);
                    return val[cur];
                }
                cur = ch[cur][1];
            }
        }
    }
    void reverse(int l,int r)
    {
        l=kth(l);r=kth(r+2);
        splay(l,0);
        splay(r,l);
        int tmp=ch[rt][1];
        mark[ch[tmp][0]]^=1;
    }
} tree;

int main()
{
    ios::sync_with_stdio(false);
    cin >> n >> m;
    for(int i=1;i<=n+2;i++)
        tree.ins(i);
    while(m--)
    {
        int l,r;
        cin>>l>>r;
        tree.reverse(l,r);
    }
    tree.write(rt);
    return 0;
}
```

## LCA

```cpp
// 倍增方法：
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#define MXN 50007
using namespace std;
std::vector<int> v[MXN];
std::vector<int> w[MXN];
int fa[MXN][31], cost[MXN][31], dep[MXN];
int n, m;
int a, b, c;
void dfs(int root, int fno) {
	fa[root][0] = fno;
	dep[root] = dep[fa[root][0]] + 1;
	for (int i = 1; i < 31; ++i) {
		fa[root][i] = fa[fa[root][i - 1]][i - 1];
		cost[root][i] = cost[fa[root][i - 1]][i - 1] + cost[root][i - 1];
	}
	int sz = v[root].size();
	for (int i = 0; i < sz; ++i) {
		if (v[root][i] == fno) continue;
		cost[v[root][i]][0] = w[root][i];
		dfs(v[root][i], root);
	}
}
int lca(int x, int y) {
	if (dep[x] > dep[y]) swap(x, y);
	int tmp = dep[y] - dep[x], ans = 0;
	for (int j = 0; tmp; ++j, tmp >>= 1)
		if (tmp & 1) ans += cost[y][j], y = fa[y][j];
	if (y == x) return ans;
	for (int j = 30; j >= 0 && y != x; --j) {
		if (fa[x][j] != fa[y][j]) {
			ans += cost[x][j] + cost[y][j];
			x = fa[x][j];
			y = fa[y][j];
		}
	}
	ans += cost[x][0] + cost[y][0];
	return ans;
}
int main() {
	memset(fa, 0, sizeof(fa));
	memset(cost, 0, sizeof(cost));
		memset(dep, 0, sizeof(dep));
	scanf("%d", &n);
	for (int i = 1; i < n; ++i) {
		scanf("%d %d %d", &a, &b, &c);
		++a, ++b;
		v[a].push_back(b);
		v[b].push_back(a);
		w[a].push_back(c);
		w[b].push_back(c);
	}
	dfs(1, 0);
	scanf("%d", &m);
	for (int i = 0; i < m; ++i) {
		scanf("%d %d", &a, &b);
		++a, ++b;
		printf("%d\n", lca(a, b));
	}
	return 0;
}
```

```cpp
// Tarjan方法：
#include <algorithm>
#include <iostream>
using namespace std;
class Edge {
public:
	int toVertex, fromVertex;
	int next;
	int LCA;
	Edge() : toVertex(-1), fromVertex(-1), next(-1), LCA(-1) {};
	Edge(int u, int v, int n) : fromVertex(u), toVertex(v), next(n), LCA(-1) {};
};
const int MAX = 100;
int head[MAX], queryHead[MAX];
Edge edge[MAX], queryEdge[MAX];
int parent[MAX], visited[MAX];
int vertexCount, edgeCount, queryCount;
void init() {
	for (int i = 0; i <= vertexCount; i++) {
		parent[i] = i;
	}
}
int find(int x) {
	if (parent[x] == x) {
		return x;
	}
	else {
		return find(parent[x]);
	}
}
void tarjan(int u) {
	parent[u] = u;
	visited[u] = 1;
	for (int i = head[u]; i != -1; i = edge[i].next) {
		Edge& e = edge[i];
		if (!visited[e.toVertex]) {
			tarjan(e.toVertex);
			parent[e.toVertex] = u;
		}
	}
	for (int i = queryHead[u]; i != -1; i = queryEdge[i].next) {
		Edge& e = queryEdge[i];
		if (visited[e.toVertex]) {
			queryEdge[i ^ 1].LCA = e.LCA = find(e.toVertex);
		}
	}
}
int main() {
	memset(head, 0xff, sizeof(head));
	memset(queryHead, 0xff, sizeof(queryHead));
	cin >> vertexCount >> edgeCount >> queryCount;
	int count = 0;
	for (int i = 0; i < edgeCount; i++) {
		int start = 0, end = 0;
		cin >> start >> end;
			edge[count] = Edge(start, end, head[start]);
		head[start] = count;
		count++;
		edge[count] = Edge(end, start, head[end]);
		head[end] = count;
		count++;
	}
	count = 0;
	for (int i = 0; i < queryCount; i++) {
		int start = 0, end = 0;
		cin >> start >> end;
		queryEdge[count] = Edge(start, end, queryHead[start]);
		queryHead[start] = count;
		count++;
		queryEdge[count] = Edge(end, start, queryHead[end]);
		queryHead[end] = count;
		count++;
	}
	init();
	tarjan(1);
	for (int i = 0; i < queryCount; i++) {
		Edge& e = queryEdge[i * 2];
		cout << "(" << e.fromVertex << "," << e.toVertex << ") " << e.LCA << endl;
	}
	return 0;
}
```

## 大数

```cpp
//注：可以直接把BigInt和一样用cin cout都行，就是高精乘为了速度才用了FFT降低了精度，有需要可以自行更改。
#include <cstdio>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace std;
const double PI = acos(-1.0);
struct Complex{
    double x,y;
    Complex(double _x = 0.0,double _y = 0.0){
        x = _x;
        y = _y;
    }
    Complex operator-(const Complex &b)const{
        return Complex(x - b.x,y - b.y);
    }
    Complex operator+(const Complex &b)const{
        return Complex(x + b.x,y + b.y);
    }
    Complex operator*(const Complex &b)const{
        return Complex(x*b.x - y*b.y,x*b.y + y*b.x);
    }
};
void change(Complex y[],int len){
    int i,j,k;
    for(int i = 1,j = len/2;i<len-1;i++){
        if(i < j)    swap(y[i],y[j]);
        k = len/2;
        while(j >= k){
            j = j - k;
            k = k/2;
        }
        if(j < k)    j+=k;
    }
}
void fft(Complex y[],int len,int on){
    change(y,len);
    for(int h = 2;h <= len;h<<=1){
        Complex wn(cos(on*2*PI/h),sin(on*2*PI/h));
        for(int j = 0;j < len;j += h){
            Complex w(1,0);
            for(int k = j;k < j + h/2;k++){
                Complex u = y[k];
                Complex t = w*y[k + h/2];
                y[k] = u + t;
                y[k + h/2] = u - t;
                w = w*wn;
            }
        }
    }
    if(on == -1){
        for(int i = 0;i < len;i++){
            y[i].x /= len;
        }
    }
}
class BigInt
{
#define Value(x, nega) ((nega) ? -(x) : (x))
#define At(vec, index) ((index) < vec.size() ? vec[(index)] : 0)
    static int absComp(const BigInt &lhs, const BigInt &rhs)
    {
        if (lhs.size() != rhs.size())
            return lhs.size() < rhs.size() ? -1 : 1;
        for (int i = lhs.size() - 1; i >= 0; --i)
            if (lhs[i] != rhs[i])
                return lhs[i] < rhs[i] ? -1 : 1;
        return 0;
    }
    using Long = long long;
    const static int Exp = 9;
    const static Long Mod = 1000000000;
    mutable std::vector<Long> val;
    mutable bool nega = false;
    void trim() const
    {
        while (val.size() && val.back() == 0)
            val.pop_back();
        if (val.empty())
            nega = false;
    }
    int size() const { return val.size(); }
    Long &operator[](int index) const { return val[index]; }
    Long &back() const { return val.back(); }
    BigInt(int size, bool nega) : val(size), nega(nega) {}
    BigInt(const std::vector<Long> &val, bool nega) : val(val), nega(nega) {}

public:
    friend std::ostream &operator<<(std::ostream &os, const BigInt &n)
    {
        if (n.size())
        {
            if (n.nega)
                putchar('-');
            for (int i = n.size() - 1; i >= 0; --i)
            {
                if (i == n.size() - 1)
                    printf("%lld", n[i]);
                else
                    printf("%0*lld", n.Exp, n[i]);
            }
        }
        else
            putchar('0');
        return os;
    }
    friend BigInt operator+(const BigInt &lhs, const BigInt &rhs)
    {
        BigInt ret(lhs);
        return ret += rhs;
    }
    friend BigInt operator-(const BigInt &lhs, const BigInt &rhs)
    {
        BigInt ret(lhs);
        return ret -= rhs;
    }
    BigInt(Long x = 0)
    {
        if (x < 0)
            x = -x, nega = true;
        while (x >= Mod)
            val.push_back(x % Mod), x /= Mod;
        if (x)
            val.push_back(x);
    }
    BigInt(const char *s)
    {
        int bound = 0, pos;
        if (s[0] == '-')
            nega = true, bound = 1;
        Long cur = 0, pow = 1;
        for (pos = strlen(s) - 1; pos >= Exp + bound - 1; pos -= Exp, val.push_back(cur), cur = 0, pow = 1)
            for (int i = pos; i > pos - Exp; --i)
                cur += (s[i] - '0') * pow, pow *= 10;
        for (cur = 0, pow = 1; pos >= bound; --pos)
            cur += (s[pos] - '0') * pow, pow *= 10;
        if (cur)
            val.push_back(cur);
    }
    BigInt &operator=(const char *s){
        BigInt n(s);
        *this = n;
        return n;
    }
    BigInt &operator=(const Long x){
        BigInt n(x);
        *this = n;
        return n;
    }
    friend std::istream &operator>>(std::istream &is, BigInt &n){
        string s;
        is >> s;
        n=(char*)s.data();
        return is;
    }
    BigInt &operator+=(const BigInt &rhs)
    {
        const int cap = std::max(size(), rhs.size()) + 1;
        val.resize(cap);
        int carry = 0;
        for (int i = 0; i < cap - 1; ++i)
        {
            val[i] = Value(val[i], nega) + Value(At(rhs, i), rhs.nega) + carry, carry = 0;
            if (val[i] >= Mod)
                val[i] -= Mod, carry = 1;
            else if (val[i] < 0)
                val[i] += Mod, carry = -1;
        }
        if ((val.back() = carry) == -1) //assert(val.back() == 1 or 0 or -1)
        {
            nega = true, val.pop_back();
            bool tailZero = true;
            for (int i = 0; i < cap - 1; ++i)
            {
                if (tailZero && val[i])
                    val[i] = Mod - val[i], tailZero = false;
                else
                    val[i] = Mod - 1 - val[i];
            }
        }
        trim();
        return *this;
    }
    friend BigInt operator-(const BigInt &rhs)
    {
        BigInt ret(rhs);
        ret.nega ^= 1;
        return ret;
    }
    BigInt &operator-=(const BigInt &rhs)
    {
        rhs.nega ^= 1;
        *this += rhs;
        rhs.nega ^= 1;
        return *this;
    }
    friend BigInt operator*(const BigInt &lhs, const BigInt &rhs)
    {
        int len=1;
        BigInt ll=lhs,rr=rhs;
        ll.nega = lhs.nega ^ rhs.nega;
        while(len<2*lhs.size()||len<2*rhs.size())len<<=1;
        ll.val.resize(len),rr.val.resize(len);
        Complex x1[len],x2[len];
        for(int i=0;i<len;i++){
            Complex nx(ll[i],0.0),ny(rr[i],0.0);
            x1[i]=nx;
            x2[i]=ny;
        }
        fft(x1,len,1);
        fft(x2,len,1);
        for(int i = 0 ; i < len; i++)
            x1[i] = x1[i] * x2[i];
        fft( x1 , len , -1 );
        for(int i = 0 ; i < len; i++)
            ll[i] = int( x1[i].x + 0.5 );
        for(int i = 0 ; i < len; i++){
            ll[i+1]+=ll[i]/Mod;
            ll[i]%=Mod;
        }
        ll.trim();
        return ll;
    }
    friend BigInt operator*(const BigInt &lhs, const Long &x){
        BigInt ret=lhs;
        bool negat = ( x < 0 );
        Long xx = (negat) ? -x : x;
        ret.nega ^= negat;
        ret.val.push_back(0);
        ret.val.push_back(0);
        for(int i = 0; i < ret.size(); i++)
            ret[i]*=xx;
        for(int i = 0; i < ret.size(); i++){
            ret[i+1]+=ret[i]/Mod;
            ret[i] %= Mod;
        }
        ret.trim();
        return ret;
    }
    BigInt &operator*=(const BigInt &rhs) { return *this = *this * rhs; }
    BigInt &operator*=(const Long &x) { return *this = *this * x; }
    friend BigInt operator/(const BigInt &lhs, const BigInt &rhs)
    {
        static std::vector<BigInt> powTwo{BigInt(1)};
        static std::vector<BigInt> estimate;
        estimate.clear();
        if (absComp(lhs, rhs) < 0)
            return BigInt();
        BigInt cur = rhs;
        int cmp;
        while ((cmp = absComp(cur, lhs)) <= 0)
        {
            estimate.push_back(cur), cur += cur;
            if (estimate.size() >= powTwo.size())
                powTwo.push_back(powTwo.back() + powTwo.back());
        }
        if (cmp == 0)
            return BigInt(powTwo.back().val, lhs.nega ^ rhs.nega);
        BigInt ret = powTwo[estimate.size() - 1];
        cur = estimate[estimate.size() - 1];
        for (int i = estimate.size() - 1; i >= 0 && cmp != 0; --i)
            if ((cmp = absComp(cur + estimate[i], lhs)) <= 0)
                cur += estimate[i], ret += powTwo[i];
        ret.nega = lhs.nega ^ rhs.nega;
        return ret;
    }
    friend BigInt operator/(const BigInt &num,const Long &x){
        bool negat = ( x < 0 );
        Long xx = (negat) ? -x : x;
        BigInt ret;
        Long k = 0;
        ret.val.resize( num.size() );
        ret.nega = (num.nega ^ negat);
        for(int i = num.size() - 1 ;i >= 0; i--){
            ret[i] = ( k * Mod + num[i]) / xx;
            k = ( k * Mod + num[i]) % xx;
        }
        ret.trim();
        return ret;
    }
    bool operator==(const BigInt &rhs) const
    {
        return nega == rhs.nega && val == rhs.val;
    }
    bool operator!=(const BigInt &rhs) const { return nega != rhs.nega || val != rhs.val; }
    bool operator>=(const BigInt &rhs) const { return !(*this < rhs); }
    bool operator>(const BigInt &rhs) const { return !(*this <= rhs); }
    bool operator<=(const BigInt &rhs) const
    {
        if (nega && !rhs.nega)
            return true;
        if (!nega && rhs.nega)
            return false;
        int cmp = absComp(*this, rhs);
        return nega ? cmp >= 0 : cmp <= 0;
    }
    bool operator<(const BigInt &rhs) const
    {
        if (nega && !rhs.nega)
            return true;
        if (!nega && rhs.nega)
            return false;
        return (absComp(*this, rhs) < 0) ^ nega;
    }
    void swap(const BigInt &rhs) const
    {
        std::swap(val, rhs.val);
        std::swap(nega, rhs.nega);
    }
};
BigInt ba,bb;
int main(){
    cin>>ba>>bb;
    cout << ba + bb << '\n';//和
    cout << ba - bb << '\n';//差
    cout << ba * bb << '\n';//积
    BigInt d;
    cout << (d = ba / bb) << '\n';//商
    cout << ba - d * bb << '\n';//余
    return 0;
}
```

## 线性基

```cpp
void Get_LB(ll x)
{
	for(int i = 62; i >= 0; i--)
	{
		if(!(x >> (ll)i))
			continue;
		if(!p[i])
		{
			p[i] = x;
			break;
		}
		x ^= p[i];
	}
}
```

## ST表

```cpp
template <typename INTEGER>
struct STMax
{
    // 从0开始
    std::vector<std::vector<INTEGER>> data;
    STMax(int siz)
    {
        int upper_pow = clz(siz) + 1;
        data.resize(upper_pow);
        data.assign(upper_pow, vector<INTEGER>());
        data[0].assign(siz, 0);
    }
    INTEGER &operator[](int where)
    {
        return data[0][where];
    }
    void generate_max()
    {
        for (auto j : range(1, data.size()))
        {
            data[j].assign(data[0].size(), 0);
            for (int i = 0; i + (1 << j) - 1 < data[0].size(); i++)
            {
                data[j][i] = std::max(data[j - 1][i], data[j - 1][i + (1 << (j - 1))]);
            }
        }
    }
    /*闭区间[l, r]，注意有效位从0开始*/
    INTEGER query_max(int l, int r)
    {
        int k = 31 - __builtin_clz(r - l + 1);
        return std::max(data[k][l], data[k][r - (1 << k) + 1]);
    }
};
```

# 莫队

```cpp
//动态
struct unit {
	int l, r, id;
  bool operator<(const unit &x) const {
    return l / sq == x.l / sq
               ? (r == x.r ? 0 : ((l / sq) & 1) ^ (r < x.r))
               : l < x.l;//莫队奇偶优化
  }
};

int totq=0,totxiu=0,sq,ans[N],ans1=0,n,m,a[N],sum[N],l=1,r=0,now=0;

int main()
{
    scanf("%d%d",&n,&m);
    for (int i=1;i<=n;i++) scanf("%d",&a[i]);
    for (int i=1;i<=m;i++)
    {
        char s[2];
        int x,y;
        scanf("%s%d%d",&s,&x,&y);
        if (s[0]=='Q')
        {
            totq++;
            q[totq].l=x; q[totq].r=y;
            q[totq].x=totxiu;
            q[totq].id=totq;
        }
        else
        {
            totxiu++;
            xiu[totxiu].id=x;
            xiu[totxiu].cl=y;
        }
    }
    sq=sqrt(n*1.0);
    sort(q+1,q+1+totq,cmp);
	for(int i=1;i<=totq;i++)
	{
        while(l>q[i].l)l--,ins(l);
        while(r<q[i].r)r++,ins(r);
		while(l<q[i].l)del(l),l++;
		while(r>q[i].r)del(r),r--;
        
		while(now<q[i].x)change(now+1),now++;
		while(now>q[i].x)change(now),now--;
		ans[q[i].id]=ans1;
	}
	for(int i=1;i<=totq;i++)
	printf("%d\n",ans[i]);
}
```

```cpp
//基本
struct node {
    ll l,r,id,A,B;
}Q[maxn];
ll be[maxn],a[maxn],sq,num,l=1,r=0,sum[maxn];
ll ans1=0;
ll gcd(ll a,ll b)
 {
 	if(b==0)return a;
 	else return gcd(b,a%b);
 }
bool cmp1(const node& A,const node&B) {return be[A.l]==be[B.l]?A.r<B.r:A.l<B.l;}
bool cmp2(const node& A,const node&B){return A.id<B.id;}
void ins(int x) {ans1=ans1+2*sum[x];sum[x]++;}
void del(int x) {ans1=ans1-2*sum[x]+2;sum[x]--;}
int main()
{
        scanf("%lld%lld",&n,&m);
        sq=sqrt(n);
        for(ll i=1;i<=n;i++)
        scanf("%d",&a[i]);
        for(ll i=1;i<=m;++i)
		{
            scanf("%d%d",&Q[i].l,&Q[i].r);
            Q[i].id=i;
        }
        for(ll i=1;i<=n;++i){
        	be[i]=i/sq+1;
		}
		sort(Q+1,Q+1+m,cmp1);
        for(int i=1;i<=m;i++){
		    while(l<Q[i].l)del(a[l]),++l;
            while(l>Q[i].l)--l,ins(a[l]);
            while(r<Q[i].r)++r,ins(a[r]);
            while(r>Q[i].r)del(a[r]),--r;
            if(Q[i].l==Q[i].r){Q[i].A=0;Q[i].B=1;continue;}
			Q[i].B=ans1;Q[i].A=(Q[i].r-Q[i].l+1)*(Q[i].r-Q[i].l);
        }
        sort(Q+1,Q+1+m,cmp2);
        for(ll i=1;i<=m;i++){
        	if(Q[i].A==0)printf("0/1\n");
        	else {
        		int g=gcd(Q[i].A,Q[i].B);
        	    printf("%lld/%lld\n",Q[i].B/g,Q[i].A/g);
			}
		}
    return 0;
}
```

# 三分

```cpp
// 整数
while(l+1<r)
{
    int lm=(l+r)>>1,rm=(lm+r)>>1;
    if(judge(lm)>judge(rm))
        r=rm;
    else
        l=lm;
}
// double

while(l+eps<r)
{
    double lm=(l+r)/2,rm=(lm+r)/2;
    if(judge(lm)>judge(rm))
        r=rm;
    else
        l=lm;
}
```
