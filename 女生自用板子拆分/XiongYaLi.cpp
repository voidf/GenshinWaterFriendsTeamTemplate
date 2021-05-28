#include "Headers.cpp"
#define XiongYaLi 1010
LL pre_match[XiongYaLi];
LL searched[M];
bool dfs(LL x, const LL searchtime)
{
    if (searched[x] == searchtime)
        return false;
    searched[x] = searchtime;
    for (LL i = hds[x]; ~i; i = E[i].next)
    {
        LL obj = E[i].to;
        if (pre_match[obj] == -1 || dfs(pre_match[obj], searchtime))
        {
            pre_match[obj] = x;
            return true;
        }
    }
    return false;
}
LL get_max_match(LL lim)
{
    LL ans;
    for (LL i = 0 + 1; i < lim + 1; i++)
        ans += dfs(i, i);
    return ans;
}