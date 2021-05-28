
#include "Headers.cpp"

#define FrontStar 666

LL edge_cnt = 0;
struct EdgeModel
{
    LL next, to, val;
} eds[FrontStar];
LL node_ctr, merged_edge_ctr, edge_ctr;
LL head[FrontStar] = {-1};

inline void add(LL u, LL v)
{
    eds[edge_cnt].to = v;
    eds[edge_cnt].next = head[u];
    head[u] = edge_cnt++;
}

inline void add(LL u, LL v, LL w)
{
    eds[edge_cnt].to = v;
    eds[edge_cnt].val = w;
    eds[edge_cnt].next = head[u];
    head[u] = edge_cnt++;
}
