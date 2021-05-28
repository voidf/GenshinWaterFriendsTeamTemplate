#include "Headers.cpp"

#define ORAFM 2333
int prime[ORAFM + 5], prime_number = 0, prv[ORAFM + 5];
// 莫比乌斯函数
int mobius[ORAFM + 5];
// 欧拉函数
LL phi[ORAFM + 5];

bool marked[ORAFM + 5];

void ORAfliter(LL MX)
{
    mobius[1] = phi[1] = 1;
    for (unsigned int i = 2; i <= MX; i++)
    {
        if (!marked[i])
        {
            prime[++prime_number] = i;
            prv[i] = i;
            phi[i] = i - 1;
            mobius[i] = -1;
        }
        for (unsigned int j = 1; j <= prime_number && i * prime[j] <= MX; j++)
        {
            marked[i * prime[j]] = true;
            prv[i * prime[j]] = prime[j];
            if (i % prime[j] == 0)
            {
                phi[i * prime[j]] = prime[j] * phi[i];
                break;
            }
            phi[i * prime[j]] = phi[prime[j]] * phi[i];
            mobius[i * prime[j]] = -mobius[i]; // 平方因数不会被处理到，默认是0
        }
    }
    // 这句话是做莫比乌斯函数和欧拉函数的前缀和
    for (unsigned int i = 2; i <= MX; ++i)
    {
        mobius[i] += mobius[i - 1];
        phi[i] += phi[i - 1];
    }
}

unordered_map<int, int> ans_mu;
unordered_map<int, LL> ans_phi;
// 杜教筛/数论分块
LL calc_phi_pref(LL x) // φ * I(常数函数) == id(恒等函数)
{
    if (x <= ORAFM)
        return phi[x];
    if (ans_phi.find(x) != ans_phi.end())
        return ans_phi[x];        // 缓存
    LL ret = (x + 1LL) * x / 2LL; // f*g即恒等函数的前缀和（等差数列求和公式
    for (unsigned int l = 2, r; l <= x; l = r + 1)
    {
        r = x / (x / l); // 跳到下取整位置
        ret -= (LL)(r - l + 1LL) * calc_phi_pref(x / l);
        // 依题意可能写成 ret -= (getsum(r) - getsum(l - 1)) * calc_mobius_pref(x / l); g的前缀和 即 getsum
    }
    return ans_phi[x] = ret;
}

int calc_mobius_pref(int x) // μ(mu, 莫比乌斯函数) * I(常数函数，恒等于1) == ε(单位函数，除n==1时为1外全为0)
{
    if (x <= ORAFM)
        return mobius[x];
    if (ans_mu.find(x) != ans_mu.end())
        return ans_mu[x];
    int ret = 1; // 单位函数的前缀就是1
    for (unsigned int l = 2, r; l <= x; l = r + 1)
    {
        r = x / (x / l);
        ret -= (r - l + 1) * calc_mobius_pref(x / l);
    }
    return ans_mu[x] = ret;
}