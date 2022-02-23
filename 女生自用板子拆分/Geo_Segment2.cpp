#ifndef Geo_Segment2_H
#define Geo_Segment2_H

#include "Geo_Base.cpp"
#include "Geo_vec2.cpp"
#include "Geo_Line2.cpp"

namespace Geometry
{

    struct Segment2 : Line2 // 二维有向线段
    {
        vec2 from, to;
        Segment2(vec2 a, vec2 b) : Line2(a, b), from(a), to(b) {}
        Segment2(fl x, fl y, fl X, fl Y) : Line2(vec2(x, y), vec2(X, Y)), from(vec2(x, y)), to(vec2(X, Y)) {}
        vec2 toward() const { return to - from; }
        /* 精度较低的判断点在线段上 */
        bool is_online(vec2 poi)
        {
            return round_compare((vec2::Distance(poi, to) + vec2::Distance(poi, from)), vec2::Distance(from, to));
        }
        /* 判断本线段的射线方向与线段b的交点会不会落在b内，认为long double可以装下long long精度，如果seg2存的点是精确的，这么判断比求交点再online更精确 */
        bool ray_in_range(const Segment2 &b) const
        {
            vec2 p = to - from;
            vec2 pl = b.to - from;
            vec2 pr = b.from - from;
            fl c1 = vec2::Cross(p, pl);
            fl c2 = vec2::Cross(p, pr);
            return c1 >= 0 and c2 <= 0 or c1 <= 0 and c2 >= 0;
        }
        /* 判断相交 */
        static bool IsIntersect(const Segment2 &u, const Segment2 &v) { return u.ray_in_range(v) && v.ray_in_range(u); }
        /* 方向向量叉积判平行，比直线判平行更精确更快，按需使用eps */
        static bool IsParallel(const Segment2 &u, const Segment2 &v)
        {
            return (vec2::Cross(u.to - u.from, v.to - v.from) == 0);
        }
        vec2 &operator[](int i)
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
        /* 防止Line2精度不足的平行线距离，一次sqrt */
        static fl Distance(const Segment2 &a, const Segment2 &b)
        {
            return a.distToPoint(b.to);
        }
        /* 点到直线的距离，一次sqrt */
        fl distToPoint(const vec2 &p) const { return abs(vec2::Cross(p - from, toward()) / toward().magnitude()); }

        /* 点到线段距离 */
        fl distToPointS(const vec2 &p) const
        {
            if (vec2::Dot(toward(), p - from) <= 0)
                return vec2::Distance(from, p);
            if (vec2::Dot(-toward(), p - to) <= 0)
                return vec2::Distance(to, p);
            return distToPoint(p);
        }

        /* 线段与线段距离 */
        fl distToSeg(const Segment2 &s) const
        {
            return min({distToPointS(s.from), distToPointS(s.to), s.distToPointS(from), s.distToPointS(to)});
        }
    };

}
#endif
