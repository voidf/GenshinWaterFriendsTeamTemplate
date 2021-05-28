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
        bool is_online(Vector2 poi)
        {
            return round_compare((Vector2::Distance(poi, to) + Vector2::Distance(poi, from)), Vector2::Distance(from, to));
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