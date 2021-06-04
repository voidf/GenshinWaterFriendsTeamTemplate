### 数位dp

```c++
int dfs(int pos, int lim1, int lim2, bool zero)
{
    if (pos== -1) return 1;
    if (dp[pos][lim1][lim2] != -1)
        return dp[pos][lim1][lim2];
    int up1 = lim1 ? a[pos] : 1;
    int up2 = lim2 ? b[pos] : 1;
    int tmp1 = 0, tmp2 = 0;
    for (int i = 0; i <= up1; i++)
        for (int j = 0; j <= up2; j++)
        {
            if (i & j)
                continue;
            int tmp = dfs(pos - 1, lim1 && (i == up1), lim2 && (j == up2), zero || (i ^ j));
            if (!zero && (i ^ j))
            {
                tmp1 = (tmp1 + tmp) % mod;
            }
            tmp2 = (tmp2 + tmp) % mod;
        }
    ans = (ans + tmp1 * (pos + 1) % mod) % mod;
    dp[pos][lim1][lim2] = tmp2;
    return tmp2;
}
void cw(int x, int y)//拆位
{
    int pos1 = 0, pos2 = 0;
    while (x)
    {
        a[pos1++] = x & 1;
        x >>= 1;
    }
    while (y)
    {
        b[pos2++] = y & 1;
        y >>= 1;
    }
    for (int i = pos1 - 1; i >= pos2; i--) b[i] = 0;
    dfs(pos1 - 1, 1, 1, 0);
}
```