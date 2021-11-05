#include "Geo_Base.cpp"
#include "Geo_Vector2.cpp"

namespace Geometry
{

    struct Polygon2
    {
        std::vector<Vector2> points;

    private:
        Vector2 accordance;

    public:
        inline Polygon2 ConvexHull()
        {
            Polygon2 ret;
            std::sort(points.begin(), points.end());
            std::vector<Vector2> &stk = ret.points;

            std::vector<char> used(points.size(), 0);
            std::vector<int> uid;
            for (auto &i : points)
            {
                while (stk.size() >= 2 and Vector2::Cross(stk.back() - stk[stk.size() - 2], i - stk.back()) <= 0)
                {
                    used[uid.back()] = 0;
                    uid.pop_back();
                    stk.pop_back();
                }

                used[&i - &points.front()] = 1;
                uid.emplace_back(&i - &points.front());
                stk.emplace_back(i);
            }
            used[0] = 0;
            int ts = stk.size();
            for (auto ii = ++points.rbegin(); ii != points.rend(); ii++)
            {
                Vector2 &i = *ii;
                if (!used[&i - &points.front()])
                {
                    while (stk.size() > ts and Vector2::Cross(stk.back() - stk[stk.size() - 2], i - stk.back()) <= 0)
                    {
                        used[uid.back()] = 0;
                        uid.pop_back();
                        stk.pop_back();
                    }
                    used[&i - &points.front()] = 1;
                    uid.emplace_back(&i - &points.front());
                    stk.emplace_back(i);
                }
            }
            stk.pop_back();
            return ret;
        }

        /* 凸多边形用逆时针排序 */
        inline void autoanticlockwiselize()
        {
            accordance = average();
            anticlockwiselize();
        }

        inline void anticlockwiselize()
        {
            auto anticlock_comparator = [&](Vector2 &a, Vector2 &b) -> bool
            {
                return (a - accordance).toPolarCoordinate(false).y < (b - accordance).toPolarCoordinate(false).y;
            };
            std::sort(points.begin(), points.end(), anticlock_comparator);
        }

        inline Vector2 average() const
        {
            Vector2 avg(0, 0);
            for (auto &i : points)
            {
                avg += i;
            }
            return avg / points.size();
        }

        /* 求周长 */
        inline FLOAT_ perimeter() const
        {
            FLOAT_ ret = Vector2::Distance(points.front(), points.back());
            for (int i = 1; i < points.size(); i++)
                ret += Vector2::Distance(points[i], points[i - 1]);
            return ret;
        }
        /* 面积 */
        inline FLOAT_ area() const
        {
            FLOAT_ ret = Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + Vector2::Cross(points[i - 1], points[i]);
            return ret / 2;
        }
        /* 求几何中心（形心、重心） */
        inline Vector2 center() const
        {
            Vector2 ret = (points.back() + points.front()) * Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + (points[i - 1] + points[i]) * Vector2::Cross(points[i - 1], points[i]);
            return ret / area() / 6;
        }
        /* 求边界整点数 */
        inline long long boundary_points() const
        {
            long long b = 0;
            for (int i = 0; i < points.size() - 1; i++)
            {
                b += std::__gcd((long long)abs(points[i + 1].x - points[i].x), (long long)abs(points[i + 1].y - points[i].y));
            }
            return b;
        }
        /* Pick定理：多边形面积=内部整点数+边界上的整点数/2-1；求内部整点数 */
        inline long long interior_points(FLOAT_ A = -1, long long b = -1) const
        {
            if (A < 0)
                A = area();
            if (b < 0)
                b = boundary_points();
            return (long long)A + 1 - (b / 2);
        }

        inline bool is_inner(const Vector2 &p) const
        {
            bool res = false;
            Vector2 j = points.back();
            for (auto &i : points)
            {
                if ((i.y < p.y and j.y >= p.y or j.y < p.y and i.y >= p.y) and (i.x <= p.x or j.x <= p.x))
                    res ^= (i.x + (p.y - i.y) / (j.y - i.y) * (j.x - i.x) < p.x);
                j = i;
            }
            return res;
        }
        /* 对接图形库的转换成vec3 float序列 */
        inline std::vector<FLOAT_> to_vec3_array() const
        {
            std::vector<FLOAT_> ret;
            ret.reserve(3 * points.size());
            for (auto &i : points)
            {
                ret.emplace_back(i.x);
                ret.emplace_back(i.y);
                ret.emplace_back(0);
            }
            return ret;
        }
        /* 极坐标割圆术返回一个细分subdivision个顶点近似的圆 */
        inline static Polygon2 cyclotomic(Vector2 center = 0, FLOAT_ radius = 1, int subdivision = 40)
        {
            Polygon2 ret;
            ret.points.reserve(subdivision);
            FLOAT_ step = 2 * PI / subdivision, cur = 0;
            while (subdivision--)
            {
                ret.points.emplace_back(center + Vector2::fromPolarCoordinate(Vector2(radius, cur), false));
                cur += step;
            }
            return ret;
        }

        /* 割圆星型 */
        inline static Polygon2 cyclotomic_star(Vector2 center = 0, FLOAT_ inner_radius = 1, FLOAT_ outer_radius = 3, int subdivision = 5)
        {
            Polygon2 ret;
            ret.points.reserve(subdivision*2);
            FLOAT_ step = 2 * PI / subdivision, cur = 0;
            while (subdivision--)
            {
                ret.points.emplace_back(center + Vector2::fromPolarCoordinate(Vector2(outer_radius, cur), false));
                ret.points.emplace_back(center + Vector2::fromPolarCoordinate(Vector2(inner_radius, cur + step/2), false));
                cur += step;
            }
            return ret;
        }
    };

}

/* 旋转卡壳用例
auto CV = P.ConvexHull();
int idx = 0;
int jdx = 1;
FLOAT_ dis = 0;
for (auto &i : CV.points)
{
    // auto cdis = (i - CV.points.front()).sqrMagnitude();
    int tj = (jdx + 1) % CV.points.size();
    int ti = (idx + 1) % CV.points.size();
    while (Vector2::Cross(CV.points[tj] - i, CV.points[ti] - i) < Vector2::Cross(CV.points[jdx] - i, CV.points[ti] - i))
    {
        jdx = tj;
        tj = (jdx + 1) % CV.points.size();
    }
    dis = max({dis, (CV.points[jdx] - i).sqrMagnitude(), (CV.points[jdx] - CV.points[ti]).sqrMagnitude()});
    
    ++idx;
}
cout << dis << endl;

*/