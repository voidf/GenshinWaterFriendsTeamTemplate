#include "Geo_Base.cpp"

namespace Geometry
{
    // https://www.luogu.com.cn/record/51596689

    struct Simpson
    {
        std::function<FLOAT_(FLOAT_)> F;
        Simpson(std::function<FLOAT_(FLOAT_)> _f) : F(_f) {}
        inline FLOAT_ simpson(FLOAT_ l, FLOAT_ r)
        {
            return (F(l) + 4 * F((l + r) / 2) + F(r)) * (r - l) / 6;
        }
        inline FLOAT_ asr(FLOAT_ l, FLOAT_ r, FLOAT_ eps, FLOAT_ ans)
        {
            FLOAT_ mid = (l + r) / 2;
            FLOAT_ _l = simpson(l, mid), _r = simpson(mid, r);
            if (abs(_l + _r - ans) <= 15 * eps)
                return (_l + _r - ans) / 15 + _l + _r;
            return asr(l, mid, eps / 2, _l) + asr(mid, r, eps / 2, _r);
        }
        FLOAT_ asr(FLOAT_ l, FLOAT_ r, FLOAT_ eps)
        {
            return asr(l, r, eps, simpson(l, r));
        }
    };

}

