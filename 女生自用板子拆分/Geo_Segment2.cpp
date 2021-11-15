#include "Geo_Base.cpp"
#include "Geo_Vector2.cpp"
#include "Geo_Line2.cpp"

namespace Geometry
{

    struct Segment2 : Line2 // 二维有向线段
    {
        Vector2 from, to;
        Segment2(Vector2 a, Vector2 b) : Line2(a, b), from(a), to(b) {}
        Segment2(FLOAT_ x, FLOAT_ y, FLOAT_ X, FLOAT_ Y) : Line2(Vector2(x, y), Vector2(X, Y)), from(Vector2(x, y)), to(Vector2(X, Y)) {}
        /* 精度较低的判断点在线段上 */
        bool is_online(Vector2 poi)
        {
            return round_compare((Vector2::Distance(poi, to) + Vector2::Distance(poi, from)), Vector2::Distance(from, to));
        }
        /* 判断本线段的射线方向与线段b的交点会不会落在b内，认为long double可以装下long long精度，如果seg2存的点是精确的，这么判断比求交点再online更精确 */
        bool ray_in_range(const Segment2 &b) const
        {
            Vector2 p = to - from;
            Vector2 pl = b.to - from;
            Vector2 pr = b.from - from;
            FLOAT_ c1 = Vector2::Cross(p, pl);
            FLOAT_ c2 = Vector2::Cross(p, pr);
            return c1 >= 0 and c2 <= 0 or c1 <= 0 and c2 >= 0;
        }
        /* 方向向量叉积判平行，比直线判平行更精确更快，按需使用eps */
        static bool parallel(const Segment2 &u, const Segment2 &v)
        {
            return (Vector2::Cross(u.to - u.from, v.to - v.from) == 0);
        }
        Vector2 &operator[](int i)
        {
            switch (i)
            {
            case 0:
                return from;
                break;
            case 1:
                return to;
                break;
            default:
                throw "数组越界";
                break;
            }
        }
    };

}