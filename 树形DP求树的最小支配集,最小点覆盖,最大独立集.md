**一:最小支配集**

考虑最小支配集,每个点有两种状态,即属于支配集合或者不属于支配集合,其中不属于支配集合时此点还需要被覆盖,被覆盖也有两种状态,即被子节点覆盖或者被父节点覆盖.总结起来就是三种状态,现对这三种状态定义如下:

1):**dp[i] [0]**,表示点 i 属于支配集合,并且以点 i 为根的子树都被覆盖了的情况下支配集中所包含最少点的个数.

2):**dp[i] [1]**,表示点 i 不属于支配集合,且以 i 为根的子树都被覆盖,且 i 被其中不少于一个子节点覆盖的情况下支配集所包含最少点的个数.

3):**dp[i] [2]**,表示点 i 不属于支配集合,且以 i 为根的子树都被覆盖,且 i 没被子节点覆盖的情况下支配集中所包含最少点的个数.即 i 将被父节点覆盖.

**对于第一种状态**,dp[i] [0]含义为点 i 属于支配集合,那么依次取每个儿子节点三种状态中的最小值,再把取得的最小值全部加起来再加 1,就是dp[i] [0]的值了.即只要每个以 i 的儿子为根的子树都被覆盖,再加上当前点 i,所需要的最少的点的个数,DP转移方程如下:

**dp[i] [0] = 1 + ∑(u 取 i 的子节点)min(dp[u] [0], dp[u] [1], dp[u] [2])**

**对于第三种状态**,dp[i] [2]含义为点 i 不属于支配集合,且 i 被其父节点覆盖.那么说明点 i 和点 i 的儿子节点都不属于支配集合,所以点 i 的第三种状态之和其儿子节点的第二种状态有关,方程为:

**dp[i] [2] = ∑(u 取 i 的子节点)dp[u] [1]**

**对于第二种状态**,略有些复杂.首先如果点 i 没有子节点那么 dp[i] [1]应该初始化为 INF;否则为了保证它的每个以 i 的儿子为根的子树被覆盖,那么要取每个儿子节点的前两种状态的最小值之和,因为此时点 i 不属于支配集,不能支配其子节点,所以子节点必须已经被支配,与子节点的第三种状态无关.如果当前所选状态中每个儿子都没被选择进入支配集,即在每个儿子的前两种状态中,第一种状态都不是所需点最小的,那么为了满足第二种状态的定义(因为点 i 的第二种状态必须被其子节点覆盖,即其子节点必须有一个属于支配集,如果此时没有,就必须强制使一个子节点的状态为状态一),需要重新选择点 i 的一个儿子节点为第一种状态,这时取花费最少的一个点,即取min(dp[u] [0] - dp[u] [1])的儿子节点 u,强制取其第一种状态,其他的儿子节点取第二种状态,DP转移方程为:

**if(i 没有子节点)  dp[i] [1] = INF**

**else          dp[i] [1] = ∑(u 取 i 的子节点)min(dp[u] [0], dp[u] [1]) + inc**

**其中对于 inc 有:**

**if(上面式子中的 ∑(u 取 i 的子节点)min(dp[u] [0], dp[u] [1]) 中包含某个 dp[u] [0], 即存在一个所选的最小值为状态一的儿子节点) inc = 0**

**else  inc = min(dp[u] [0] - dp[u] [1]) (其中 u 取点 i 的儿子节点)**

**代码:**

```
 1 void DP(int u, int p) {// p 为 u 的父节点
 2     dp[u][2] = 0;
 3     dp[u][0] = 1;
 4     bool s = false;
 5     int sum = 0, inc = INF;
 6     for(int k = head[u]; k != -1; k = edge[k].next) {
 7         int to = edge[k].to;
 8         if(to == p) continue;
 9         DP(to, u);
10         dp[u][0] += min(dp[to][0], min(dp[to][1], dp[to][2]));
11         if(dp[to][0] <= dp[to][1]) {
12             sum += dp[to][0];
13             s = true;
14         }
15         else {
16             sum += dp[to][1];
17             inc = min(inc, dp[to][0] - dp[to][1]);
18         }
19         if(dp[to][1] != INF && dp[u][2] != INF) dp[u][2] += dp[to][1];
20         else dp[u][2] = INF;
21     }
22     if(inc == INF && !s) dp[u][1] = INF;
23     else {
24         dp[u][1] = sum;
25         if(!s) dp[u][1] += inc;
26     }
27 }
```

**二:最小点覆盖**

对于最小点覆盖,每个点只有两种状态,即属于点覆盖或者不属于点覆盖:

1):**dp[i] [0]**表示点 i 属于点覆盖,并且以点 i 为根的子树中所连接的边都被覆盖的情况下点覆盖集中所包含最少点的个数.

2):**dp[i] [1]**表示点 i 不属于点覆盖,且以点 i 为根的子树中所连接的边都被覆盖的情况下点覆盖集中所包含最少点的个数.

**对于第一种状态**dp[i] [0],等于每个儿子节点的两种状态的最小值之和加 1,DP转移方程如下:

**dp[i] [0] = 1 + ∑(u 取 i 的子节点)min(dp[u] [0], dp[u] [1])**

**对于第二种状态**dp[i] [1],要求所有与 i 连接的边都被覆盖,但是点 i 不属于点覆盖,那么点 i 的所有子节点就必须属于点覆盖,即对于点 i 的第二种状态与所有子节点的第一种状态有关,在数值上等于所有子节点第一种状态的和.DP转移方程如下:

**dp[i] [1] = ∑(u 取 i 的子节点)dp[u] [0]
**

**代码：**

```
 1 void DP(int u, int p) {// p 为 u 的父节点
 2     dp[u][0] = 1;
 3     dp[u][1] = 0;
 4     for(int k = head[u]; k != -1; k = edge[k].next) {
 5         int to = edge[k].to;
 6         if(to == p) continue;
 7         DP(to, u);
 8         dp[u][0] += min(dp[to][0], dp[to][1]);
 9         dp[u][1] += dp[to][0];
10     }
11 }
```

**三:最大独立集**

对于最大独立集,每个点也只有两种状态,即属于点 i 属于独立集或者不属于独立集两种情况:

1):**dp[i] [0]**表示点 i 属于独立集的情况下,最大独立集中点的个数.

2):**dp[i] [1]**表示点 i 不属于独立集的情况下.最大独立集中点的个数.

**对于第一种状态**dp[i] [0],由于 i 点属于独立集,所以它的子节点都不能属于独立集,所以对于点 i 的第一种状态,只和它的子节点的第二种状态有关.等于其所有子节点的第二种状态的值加 1,DP转移方程如下:

**dp[i] [0] = 1 + ∑(u 取 i 的子节点) dp[u] [1]**

**对于第二种状态**dp[i] [1],由于点 i 不属于独立集,所以子节点可以属于独立解,也可以不属于独立集,所取得子节点状态应该为数值较大的那个状态,DP转移方程:

**dp[i] [1] = ∑(u 取 i 的子节点)max(dp[u] [0], dp[u] [1])**

**代码:**

```
void DP(int u, int p) {// p 为 u 的父节点
    dp[u][0] = 1;
    dp[u][1] = 0;
    for(int k = head[u]; k != -1; k = edge[k].next) {
        int to = edge[k].to;
        if(to == p) continue;
        DP(to, u);
        dp[u][0] += dp[to][1];
        dp[u][1] += max(dp[to][0], dp[to][1]);
    }
}
```