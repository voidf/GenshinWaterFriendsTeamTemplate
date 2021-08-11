#include "Headers.cpp"

using FT = long double;

FT fun(FT angle) // 根据需要改 评估函数
{
    FT res = 0;
    for (auto &[TT, SS, AA] : V)
    {
        FT deg = abs(angle - AA);
        res += max(FT(0.0), TT - SS * (deg >= pi ? oneround - deg : deg));
    }

    return res;
}



void sa(FT temperature = 300, FT cooldown = 1e-14, FT cool = 0.986)
{
    FT cangle = randreal(0, oneround);
    FT jbj = fun(cangle); // 局部解
    MX = max(MX, jbj); // 全局解

    while (temperature > cooldown) 
    {
        FT curangle = fmod(cangle + randreal(-1, 1) * temperature, oneround);
        while (curangle < 0)
            curangle += oneround;

        FT energy = fun(curangle);
        FT de = jbj - energy;
        MX = max(jbj, MX);
        if (de < 0)
        {
            cangle = curangle;
            jbj = energy;
        }
        else if (exp(-de / (temperature)) > randreal(0, 1))
        {
            cangle = curangle;
            jbj = energy;
        }
        temperature *= cool;
    }
}