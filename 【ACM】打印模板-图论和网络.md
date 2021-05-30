# 图论

## Floyd

```cpp
for (k = 1; k <= n; k++) {
	for (i = 1; i <= n; i++) {
		for (j = 1; j <= n; j++) {
			f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
		}
	}
}
```

## SPFA（它已经死了    //vector？

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxN = 200010;
struct Edge
{
    int to, next, w;
} e[maxN];
int n, m, cnt, p[maxN], Dis[maxN];
int In[maxN];
bool visited[maxN];
void Add_Edge(const int x, const int y, const int z)
{
    e[++cnt].to = y;
    e[cnt].next = p[x];
    e[cnt].w = z;
    p[x] = cnt;
    return;
}
bool Spfa(const int S)
{
    int i, t, temp;
    queue<int> Q;
    memset(visited, 0, sizeof(visited));
    memset(Dis, 0x3f, sizeof(Dis));
    memset(In, 0, sizeof(In));
    Q.push(S);
    visited[S] = true;
    Dis[S] = 0;
    while (!Q.empty())
    {
        t = Q.front();
        Q.pop();
        visited[t] = false;
        for (i = p[t]; i; i = e[i].next)
        {
            temp = e[i].to;
            if (Dis[temp] > Dis[t] + e[i].w)
            {
                Dis[temp] = Dis[t] + e[i].w;
                if (!visited[temp])
                {
                    Q.push(temp);
                    visited[temp] = true;
                    if (++In[temp] > n)
                        return false;
                }
            }
        }
    }
    return true;
}
int main()
{
    int S, T;
    scanf("%d%d%d%d", &n, &m, &S, &T);
    for (int i = 1; i <= m; ++i)
    {
        int x, y, _;
        scanf("%d%d%d", &x, &y, &_);
        Add_Edge(x, y, _);
    }
    if (!Spfa(S))
        printf("FAIL!\n");
    else
        printf("%d\n", Dis[T]);
    return 0;
}
```

## Dijkstra  //vector？

```cpp
#include <cstdio>
#include <queue>
#include <vector>
#define MAXN 200010
#define INF 0x3fffffff
using namespace std;
struct edge {
    int v, w;
    edge(int v, int w) :v(v), w(w) {}
};
vector <edge> mp[MAXN];
int dis[MAXN];
bool vis[MAXN];
int n, m, s;
struct node {
    int v, dis;
    node(int v, int dis) :v(v), dis(dis) {}
    const bool operator < (const node& a) const {
        return a.dis < dis;
    }
};
priority_queue <node> q;

int read() {
    char ch; int s = 0;
    ch = getchar();
    while (ch < '0' || ch>'9') ch = getchar();
    while (ch >= '0' && ch <= '9') s = s * 10 + ch - '0', ch = getchar();
    return s;
}

void dj() {
    for (register int i = 1; i <= n; i++)
        dis[i] = INF;
    dis[s] = 0;
    q.push(node(s, 0));
    while (!q.empty()) {
        node cur = q.top();
        q.pop();
        if (vis[cur.v])  continue;
        vis[cur.v] = 1;
        for (register int i = 0; i < mp[cur.v].size(); i++) {
            edge to = mp[cur.v][i];
            if (vis[to.v]) continue;
            if (dis[to.v] > to.w + dis[cur.v]) {
                dis[to.v] = to.w + dis[cur.v], q.push(node(to.v, dis[to.v]));
            }
        }
    }
    for (register int i = 1; i <= n; i++)
        printf("%d ", dis[i]);
}
int main()
{
    n = read(), m = read(), s = read();
    for (register int i = 1; i <= m; i++) {
        int u, v, w;
        u = read(), v = read(), w = read();
        mp[u].push_back(edge(v, w));
    }
    dj();
    return 0;
}
```

# Tarjan

## 强连通分量

```c++

const int MAXN = 1e5 + 10;
struct Edge{
	int to, next, dis;
}edge[MAXN << 1];
int head[MAXN], cnt, ans; 
bool inStack[MAXN]; 	//判断是否在栈中 
//dfn 第一次访问到该节点的时间（时间戳）
//low[i] low[i]能从哪个点（最早时间戳）到达这个点的。 
int dfn[MAXN], low[MAXN], tot;	
stack<int> stc;
void add_edge(int u, int v, int dis) {
	edge[++cnt].to = v;
	edge[cnt].next = head[u];
	head[u] = cnt;
}
void Tarjan(int x) {
	dfn[x] = low[x] = ++tot;
	stc.push(x);
	inStack[x] = 1;
	for(int i = head[x]; i; i = edge[i].next) {
		int to = edge[i].to;
		if ( !dfn[to] ) {
			Tarjan(to);
			low[x] = min(low[x], low[to]);
		} else if (inStack[to]){
			low[x] = min(low[x], dfn[to]);
		}
	}
	//cout << x << " " << low[x] << " " << dfn[x] << endl;
	if(low[x] == dfn[x]) {	//发现是整个强连通分量子树里 的最小根。
		//int cnt = 0;
		ans++;	//强连通分量计数器 
		while(1) {
			int top = stc.top();
			stc.pop();
			//cnt ++;
			inStack[top] = 0;
			//cout << top << " "; 每个强连通分量内的点 
			if(top == x) break;
		} 
	}
}
void init() {
	cnt = 1;
	tot = 0;
	ans = 0;
	memset(inStack, 0, sizeof(inStack));
	memset(head, 0, sizeof(head));
	memset(dfn, 0, sizeof(dfn));
	memset(low, 0, sizeof(low));
	while(!stc.empty()) stc.pop();
}
int main () {
	std::ios::sync_with_stdio(false);
	cin.tie(0);
	int n, m;
	while(cin >> n >> m && (n || m)){
		init();
		int x, y;
		for(int i = 1; i <= m; ++i) {
			cin >> x >> y;
			add_edge(x, y, 0);	//有向图求强连通 
		} 
		for(int i = 1; i <= n; ++i) {
			if( !dfn[i] )
				Tarjan(i);
		} 
	}
	return 0;
}
```



## 缩点

```cpp
const int MAXN = 5e3 + 20;
const int MAXM = 1e6 + 10; 
int head[MAXN], cnt, tot, dfn[MAXN], low[MAXN], color[MAXN], col;
bool vis[MAXN];
int degree[MAXN];
stack<int> stc;
int n, m;
struct Edge {
	int to, next, dis;	
}edge[MAXM << 1];
void add_edge(int u, int v, int dis) {
	edge[++cnt].to = v;
	edge[cnt].next = head[u];
	head[u] = cnt; 
}
void Tarjan(int x) {
	vis[x] = 1;
	dfn[x] = low[x] = ++tot;
	stc.push(x);
	for(int i = head[x]; i; i = edge[i].next) {
		int to = edge[i].to;
		if (!dfn[to]) {
			Tarjan(to);
			low[x] = min(low[x], low[to]);
		} else if( vis[to] ) {
			low[x] = min(low[x], dfn[to]);
		}
	}
	if(dfn[x] == low[x]) {
		col ++;
		while(true) {
			int top = stc.top();
			stc.pop();
			color[top] = col;	//颜色相同的点缩点 
			vis[top] = 0;
		//	cout << top << " ";
			if(top == x) break; 
		}
		//cout << endl;
	}
}
void solve(){
	for(int i = 1; i <= n; ++i) {
		if(!dfn[i])
			Tarjan(i);
	}
	
	for(int x = 1; x <= n; ++x) {	//遍历 n个节点 
		for(int i = head[x]; i; i = edge[i].next) {	//缩点后  每个点的出度 
			int to = edge[i].to;
			if(color[x] != color[to]) {
				degree[color[x]] ++;
			}
		}
	}
}
void init () {
	cnt = 1;
	tot = 0;
	col = 0;
	memset(vis, 0, sizeof(vis));
	memset(head, 0, sizeof(head));
	memset(dfn, 0, sizeof(dfn));
	memset(low, 0, sizeof(low));
	memset(degree, 0, sizeof(degree));
	memset(color, 0, sizeof(color));
	while(!stc.empty()) stc.pop();
}
int main () {
	std::ios::sync_with_stdio(false);
	cin.tie(0);
	while(cin >> n && n) {
		cin >> m;
		init();
		int x, y;
		for(int i = 1; i <= m; ++i) {
			cin >> x >> y;
			add_edge(x, y, 0);
		}
		solve();
	} 
	return 0;
}
```
## 割点

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m; // n：点数 m：边数
int num[100001], low[100001], inde, res;
// num：记录每个点的时间戳
// low：能不经过父亲到达最小的编号，inde：时间戳，res：答案数量
bool vis[100001], flag[100001]; // flag: 答案 vis：标记是否重复
vector<int> edge[100001]; // 存图用的
void Tarjan(int u, int father) { // u 当前点的编号，father 自己爸爸的编号
	vis[u] = true; // 标记
	low[u] = num[u] = ++inde; // 打上时间戳
	int child = 0; // 每一个点儿子数量
	for (auto v : edge[u]) { // 访问这个点的所有邻居 （C++11）
		if (!vis[v]) {
			child++; // 多了一个儿子
			Tarjan(v, u); // 继续
			low[u] = min(low[u], low[v]); // 更新能到的最小节点编号
			if (father != u && low[v] >= num[u] &&!flag[u]) // 主要代码
				// 如果不是自己，且不通过父亲返回的最小点符合割点的要求，并且没有被标记过
				// 要求即为：删了父亲连不上去了，即为最多连到父亲
			{
				flag[u] = true;
				res++; // 记录答案
			}
		}
		else if (v != father)
			low[u] =
			min(low[u], num[v]); // 如果这个点不是自己，更新能到的最小节点编号
	}
	if (father == u && child >= 2 &&
		!flag[u]) { // 主要代码，自己的话需要 2 个儿子才可以
		flag[u] = true;
		res++; // 记录答案
	}
}
int main() {
	cin >> n >> m; // 读入数据
	for (int i = 1; i <= m; i++) { // 注意点是从 1 开始的
		int x, y;
		cin >> x >> y;
		edge[x].push_back(y);
		edge[y].push_back(x);
	} // 使用 vector 存图
	for (int i = 1; i <= n; i++) // 因为 Tarjan 图不一定连通
		if (!vis[i]) {
			inde = 0; // 时间戳初始为 0
			Tarjan(i, i); // 从第 i 个点开始，父亲为自己
		}
	cout << res << endl;
	for (int i = 1; i <= n; i++)
		if (flag[i]) cout << i << " "; // 输出结果
	return 0;
}
```
## 点双连通分量

```c++

#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll mod=998244353;
const int maxn=1e6+50;
const ll inf=0x3f3f3f3f3f3f3f3fLL;
 
struct Edge{
    int u,v;
};
///割顶 bccno 无意义
int pre[maxn],iscut[maxn],bccno[maxn],dfs_clock,bcc_cut;
vector<int>G[maxn],bcc[maxn];
stack<Edge>S;
int dfs(int u,int fa){
    int lowu = pre[u] = ++dfs_clock;
    int child = 0;
    for(int i = 0; i < G[u].size(); i++){
        int v =G[u][i];
        Edge e = (Edge){u,v};
        if(!pre[v]){ ///没有访问过
            S.push(e);
            child++;
            int lowv = dfs(v, u);
            lowu=min(lowu, lowv);                ///用后代更新
            if(lowv >= pre[u]){
                iscut[u]=true;
                bcc_cut++;bcc[bcc_cut].clear();   ///注意 bcc从1开始
                for(;;){
                    Edge x=S.top();S.pop();
                    if(bccno[x.u] != bcc_cut){bcc[bcc_cut].push_back(x.u);bccno[x.u]=bcc_cut;}
                    if(bccno[x.v] != bcc_cut){bcc[bcc_cut].push_back(x.v);bccno[x.v]=bcc_cut;}
                    if(x.u==u&&x.v==v)break;
                }
            }
        }
        else if(pre[v] < pre[u] && v !=fa){
            S.push(e);
            lowu = min(lowu,pre[v]);
        }
    }
    if(fa < 0 && child == 1) iscut[u] = 0;
    return lowu;
}
void find_bcc(int n){
    memset(pre, 0, sizeof(pre));
    memset(iscut, 0, sizeof(iscut));
    memset(bccno, 0, sizeof(bccno));
    dfs_clock = bcc_cut = 0;
    for(int i = 0; i < n;i++)
        if(!pre[i])dfs(i,-1);
}
```

## 边双连通分量

去除所有桥dfs即可

## 桥：

```cpp
int low[MAXN], dfn[MAXN], iscut[MAXN], dfs_clock;
bool isbridge[MAXN];
vector<int> G[MAXN];
int cnt_bridge;
int father[MAXN];
void tarjan(int u, int fa) {
	father[u] = fa;
	low[u] = dfn[u] = ++dfs_clock;
	for (int i = 0; i < G[u].size(); i++) {
		int v = G[u][i];
		if (!dfn[v]) {
			tarjan(v, u);
			low[u] = min(low[u], low[v]);
			if (low[v] > dfn[u]) {//主要
				isbridge[v] = true;
				++cnt_bridge;
			}
		}
		else if (dfn[v] < dfn[u] && v != fa) {
			low[u] = min(low[u], dfn[v]);
		}
	}
}
```

## 2-SAT

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int maxn=1e6+5;
int a[maxn<<1];
vector<int>g[maxn<<1];
int tot;
int dfn[maxn<<1],low[maxn<<1];
stack<int>sta;
int insta[maxn<<1];
int scccnt;
int color[maxn<<1];
int n,m;
void tarjan(int u)
{
    dfn[u]=low[u]=++tot;
    sta.push(u);
    insta[u]=1;
    for(auto v:g[u])
    {
        if(!dfn[v])
        {
            tarjan(v);
            low[u]=min(low[u],low[v]);
        }
        else if(insta[v])
        {
            low[u]=min(low[u],dfn[v]);
        }
    }
    if(dfn[u]==low[u])
    {
        ++scccnt;
        do{
            color[u]=scccnt;
            u=sta.top();sta.pop();
            insta[u]=0;
        }while(low[u]!=dfn[u]);
    }
    
}
signed main()
{
    ios::sync_with_stdio(false);
  	
    cin>>n>>m;
    for(int i=1;i<=m;i++)
    {
        int a,va,b,vb;//a为va或者b为vb
        cin>>a>>va>>b>>vb;
        if(va&vb)
        {
            g[a+n].push_back(b);
            g[b+n].push_back(a);
        }
        else if(!va&vb)
        {
            g[a].push_back(b);
            g[b+n].push_back(a+n);
        }
        else if(va&!vb)
        {
            g[a+m].push_back(b+n);
            g[b].push_back(a);
        }
        else if(!va&!vb)
        {
            g[a].push_back(b+n);
            g[b].push_back(a+n);
        }
    }
    /*for (int i = 0; i < m; ++i) {
    int a = read(), va = read(), b = read(), vb = read();
    g[a + n * (va & 1)].push_back(b + n * (vb ^ 1));
    g[b + n * (vb & 1)].push_back(a + n * (va ^ 1));*/
    for(int i=1;i<=(n<<1);i++)
    {
        if(!dfn[i])
            tarjan(i);
    }
    for(int i=1;i<=n;i++)
    {
        if(color[i]==color[i+n])
        {
            cout<<"IMPOSSIBLE\n";
            return 0;
        }
    }
    cout<<"POSSIBLE\n";
    for(int i=1;i<=n;i++)
    {
        int tmp=(color[i]<color[i+n])?1:0;
        cout<<tmp<<" ";
    }
}
```

## 找环

```cpp
void tarjan(int u) {
    low[u] = dfn[u] = ++dfsClock;
    stk.push(u); ins[u] = true;
    for (const auto &v : g[u]) {
        if (!dfn[v]) tarjan(v), low[u] = std::min(low[u], low[v]);
        else if (ins[v]) low[u] = std::min(low[u], dfn[v]);
    }
    if (low[u] == dfn[u]) {
        ++sccCnt;
        do {
            color[u] = sccCnt;
            u = stk.top(); stk.pop(); ins[u] = false;
        } while (low[u] != dfn[u]);
    }
}
// 笔者使用了 Tarjan 找环，得到的 color[x] 是 x 所在的 scc 的拓扑逆序。
for (int i = 1; i <= (n << 1); ++i) if (!dfn[i]) tarjan(i);
for (int i = 1; i <= n; ++i)
    if (color[i] == color[i + n]) { // x 与 -x 在同一强连通分量内，一定无解
        puts("IMPOSSIBLE");
        exit(0);
    }
puts("POSSIBLE");
for (int i = 1; i <= n; ++i)
    print((color[i] < color[i + n])), putchar(' '); // 如果不使用 Tarjan 找环，请改成大于号

```

# 网络流

## 最大流-Dinic

```cpp
LL distant[M]; // 用于分层图层次表示
LL current[M]; // 当前弧优化
LL n, m, src, dst;

inline LL bfs()
{
	for (auto i = 1; i <= n; i++)
		distant[i] = INF;
	queue<LL> Q;
	Q.push(src);
	distant[src] = 0;
	current[src] = hds[src];
	while (!Q.empty())
	{
		auto x = Q.front();
		Q.pop();
		for (auto i = hds[x]; ~i; i = E[i].next)
		{
			auto v = E[i].to;
			if (E[i].val > 0 and distant[v] == INF)
			{
				Q.push(v);
				current[v] = hds[v];
				distant[v] = distant[x] + 1;
				if (v == dst)
					return 1;
			}
		}
	}
	return 0;
}

inline LL dfs(LL x, LL Sum)
{
	if (x == dst)
		return Sum;
	LL res = 0;
	for (auto i = current[x]; ~i and Sum; i = E[i].next) // 当前弧优化：改变枚举起点
	{
		current[x] = i;
		auto v = E[i].to;
		if (E[i].val > 0 and (distant[v] == distant[x] + 1))
		{
			LL remain = dfs(v, min(Sum, E[i].val)); // remain:当前最小剩余的流量
			if (remain == 0)
				distant[v] = INF; // 去掉增广完毕的点
			E[i].val -= remain;
			E[i ^ 1].val += remain;
			res += remain; // 经过该点的所有流量和
			Sum -= remain; // 该点剩余流量
		}
	}
	return res;
}

LL Dinic()
{
	LL ans = 0;
	while (bfs())
		ans += dfs(src, INF);
	return ans;
}
```

## 最大流-HLPP+黑魔法优化

```cpp
/* 除非卡时不然别用的预流推进桶排序优化黑魔法，用例如下
signed main()
{
	qr(HLPP::n);
	qr(HLPP::m);
	qr(HLPP::src);
	qr(HLPP::dst);
	while (HLPP::m--)
	{
		LL t1, t2, t3;
		qr(t1);
		qr(t2);
		qr(t3);
		HLPP::add(t1, t2, t3);
	}
	cout << HLPP::hlpp(HLPP::n + 1, HLPP::src, HLPP::dst) << endl;
	return 0;
}
*/
namespace HLPP
{
	const LL INF = 0x3f3f3f3f3f3f;
	const LL MXn = 1203;
	const LL maxm = 520010;

	vector<LL> gap;
	LL n, m, src, dst, now_height, src_height;

	struct NODEINFO
	{
		LL height = MXn, traffic;
		LL getIndex();
		NODEINFO(LL h = 0) : height(h) {}
		bool operator<(const NODEINFO &a) const { return height < a.height; }
	} node[MXn];

	LL NODEINFO::getIndex() { return this - node; }

	struct EDGEINFO
	{
		LL to;
		LL flow;
		LL opposite;
		EDGEINFO(LL a, LL b, LL c) : to(a), flow(b), opposite(c) {}
	};
	std::list<NODEINFO *> dlist[MXn];
	vector<std::list<NODEINFO *>::iterator> iter;
	vector<NODEINFO *> list[MXn];
	vector<EDGEINFO> edge[MXn];

	inline void add(LL u, LL v, LL w = 0)
	{
		edge[u].push_back(EDGEINFO(v, w, (LL)edge[v].size()));
		edge[v].push_back(EDGEINFO(u, 0, (LL)edge[u].size() - 1));
	}

	priority_queue<NODEINFO> PQ;
	inline bool prework_bfs(NODEINFO &src, NODEINFO &dst, LL &n)
	{
		gap.assign(n, 0);
		for (auto i = 0; i <= n; i++)
			node[i].height = n;
		dst.height = 0;
		queue<NODEINFO *> q;
		q.push(&dst);
		while (!q.empty())
		{
			NODEINFO &top = *(q.front());
			for (auto i : edge[&top - node])
			{
				if (node[i.to].height == n and edge[i.to][i.opposite].flow > 0)
				{
					gap[node[i.to].height = top.height + 1]++;
					q.push(&node[i.to]);
				}
			}
			q.pop();
		}

		return src.height == n;
	}

	inline void relabel(NODEINFO &src, NODEINFO &dst, LL &n)
	{
		prework_bfs(src, dst, n);
		for (auto i = 0; i <= n; i++)
			list[i].clear(), dlist[i].clear();

		for (auto i = 0; i <= n; i++)
		{
			NODEINFO &u = node[i];
			if (u.height < n)
			{
				iter[i] = dlist[u.height].insert(dlist[u.height].begin(), &u);
				if (u.traffic > 0)
					list[u.height].push_back(&u);
			}
		}
		now_height = src_height = src.height;
	}

	inline bool push(NODEINFO &u, EDGEINFO &dst) // 从x到y尽可能推流，p是边的编号
	{
		NODEINFO &v = node[dst.to];
		LL w = min(u.traffic, dst.flow);
		dst.flow -= w;
		edge[dst.to][dst.opposite].flow += w;
		u.traffic -= w;
		v.traffic += w;
		if (v.traffic > 0 and v.traffic <= w)
			list[v.height].push_back(&v);
		return u.traffic;
	}

	inline void push(LL n, LL ui)
	{
		auto new_height = n;
		NODEINFO &u = node[ui];
		for (auto &i : edge[ui])
		{
			if (i.flow)
			{
				if (u.height == node[i.to].height + 1)
				{
					if (!push(u, i))
						return;
				}
				else
					new_height = min(new_height, node[i.to].height + 1); // 抬到正好流入下一个点
			}
		}
		auto height = u.height;
		if (gap[height] == 1)
		{
			for (auto i = height; i <= src_height; i++)
			{
				for (auto it : dlist[i])
				{
					gap[(*it).height]--;
					(*it).height = n;
				}
				dlist[i].clear();
			}
			src_height = height - 1;
		}
		else
		{
			gap[height]--;
			iter[ui] = dlist[height].erase(iter[ui]);
			u.height = new_height;
			if (new_height == n)
				return;
			gap[new_height]++;
			iter[ui] = dlist[new_height].insert(dlist[new_height].begin(), &u);
			src_height = max(src_height, now_height = new_height);
			list[new_height].push_back(&u);
		}
	}

	inline LL hlpp(LL n, LL s, LL t)
	{
		if (s == t)
			return 0;
		now_height = src_height = 0;
		NODEINFO &src = node[s];
		NODEINFO &dst = node[t];
		iter.resize(n);
		for (auto i = 0; i < n; i++)
			if (i != s)
				iter[i] = dlist[node[i].height].insert(dlist[node[i].height].begin(), &node[i]);
		gap.assign(n, 0);
		gap[0] = n - 1;
		src.traffic = INF;
		dst.traffic = -INF; // 上负是为了防止来自汇点的推流
		for (auto &i : edge[s])
			push(src, i);
		src.traffic = 0;
		relabel(src, dst, n);
		for (LL ui; now_height >= 0;)
		{
			if (list[now_height].empty())
			{
				now_height--;
				continue;
			}
			NODEINFO &u = *(list[now_height].back());
			list[now_height].pop_back();
			push(n, &u - node);
		}
		return dst.traffic+INF;
	}
} 
```

## 最小费用最大流（MCMF）Zkw

```cpp
template <typename T>
struct MCMF // 费用流(Dinic)zkw板子
{           // Based on Dinic (zkw)

    typedef long long LL;
    T INF;
    int N = 1e5 + 5; // 最大点meta参数，要按需改
#define _N 100006
    std::bitset<_N> vis; // 要一起改
    T *Dis;
    int s, t;           // 源点，汇点需要外部写入
    int *Cur;           // 当前弧优化用
    T maxflow, mincost; // 放最终答案

    struct EdgeContent
    {
        int to;
        T flow;
        T cost;
        int dualEdge;
        EdgeContent(int a, T b, T c, int d) : to(a), flow(b), cost(c), dualEdge(d) {}
    };

    std::vector<EdgeContent> *E;

    MCMF(int n)
    {
        N = n;
        E = new std::vector<EdgeContent>[n + 1];
        Dis = new T[n + 1];
        Cur = new int[n + 1];
        maxflow = mincost = 0;
        memset(&INF, 0x3f, sizeof(INF));
    }

    void add(int u, int v, T f, T w) // 加一条u到v流为f单位费为w的边
    {
        E[u].emplace_back(v, f, w, E[v].size());
        E[v].emplace_back(u, 0, -w, E[u].size() - 1);
    }

    bool SPFA()
    {
        std::queue<int> Q;
        Q.emplace(s);
        memset(Dis, INF, sizeof(T) * (N + 1));
        Dis[s] = 0;
        int k;
        while (!Q.empty())
        {
            k = Q.front();
            Q.pop();
            vis.reset(k);
            for (auto [to, f, w, rev] : E[k])
            {
                if (f and Dis[k] + w < Dis[to])
                {
                    Dis[to] = Dis[k] + w;
                    if (!vis.test(to))
                    {
                        Q.emplace(to);
                        vis.set(to);
                    }
                }
            }
        }
        return Dis[t] != INF;
    }
    T DFS(int k, T flow)
    {
        if (k == t)
        {
            maxflow += flow;
            return flow;
        }
        T sum = 0;
        vis.set(k);
        for (auto i = Cur[k]; i < E[k].size(); i++)
        {
            auto &[to, f, w, rev] = E[k][i];
            if (!vis.test(to) and f and Dis[to] == Dis[k] + w)
            {
                Cur[k] = i;
                T p = DFS(to, std::min(flow - sum, f));
                sum += p;
                f -= p;
                E[to][rev].flow += p;
                mincost += p * w;
                if (sum == flow)
                    break;
            }
        }
        vis.reset(k);
        return sum;
    }

    void Dinic() // 入口
    {
        while (SPFA())
        {
            memset(Cur, 0, sizeof(int) * (N + 1));
            DFS(s, INF);
        }
    }
};
```