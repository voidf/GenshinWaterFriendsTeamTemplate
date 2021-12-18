#ifndef Geo_Simpson_H
#define Geo_Simpson_H

#include "Geo_Base.cpp"

namespace Geometry
{
    // https://www.luogu.com.cn/record/51596689

    struct Simpson
    {
        std::function<fl(fl)> F;
        Simpson(std::function<fl(fl)> _f) : F(_f) {}
        inline fl simpson(fl l, fl r)
        {
            return (F(l) + 4 * F((l + r) / 2) + F(r)) * (r - l) / 6;
        }
        inline fl adaptive(fl l, fl r, fl eps, fl ans)
        {
            fl mid = (l + r) / 2;
            fl _l = simpson(l, mid), _r = simpson(mid, r);
            if (abs(_l + _r - ans) <= 15 * eps)
                return (_l + _r - ans) / 15 + _l + _r;
            return adaptive(l, mid, eps / 2, _l) + adaptive(mid, r, eps / 2, _r);
        }
        fl adaptive(fl l, fl r, fl eps)
        {
            return adaptive(l, r, eps, simpson(l, r));
        }
    };

}
#endif
