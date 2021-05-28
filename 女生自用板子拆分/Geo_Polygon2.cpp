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
        Polygon2 ConvexHull()
        {
            Polygon2 ret;
            sort(points.begin(), points.end());
            std::vector<Vector2> &stk = ret.points;

            std::vector<char> used(points.size(), 0);
            std::vector<int> uid;
            for (auto &i : points)
            {
                while (stk.size() >= 2 and Vector2::Cross(stk.back() - stk[stk.size() - 2], i - stk.back()) <= 0)
                {
                    // if (stk.size() >= 2)
                    // {
                    //     auto c = Vector2::Cross(stk.back() - stk[stk.size() - 2], i - stk.back());
                    //     cerr << "c:" << c << endl;
                    // }
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

        /*凸多边形用逆时针排序*/
        void autoanticlockwiselize()
        {
            accordance = average();
            anticlockwiselize();
        }

        // typedef bool(Polygon2::*comparator);

        void anticlockwiselize()
        {
            // comparator cmp = &Polygon2::anticlock_comparator;
            auto anticlock_comparator = [&](Vector2 &a, Vector2 &b) -> bool {
                return (a - accordance).toPolarCoordinate(false).y < (b - accordance).toPolarCoordinate(false).y;
            };
            sort(points.begin(), points.end(), anticlock_comparator);
            // for (auto i : points)
            // {
            //     cerr << (i - accordance).toPolarCoordinate() << "\t" << i << endl;
            // }
        }

        Vector2 average()
        {
            Vector2 avg(0, 0);
            for (auto i : points)
            {
                avg += i;
            }
            return avg / points.size();
        }

        /*求周长*/
        FLOAT_ perimeter()
        {
            FLOAT_ ret = Vector2::Distance(points.front(), points.back());
            for (int i = 1; i < points.size(); i++)
                ret += Vector2::Distance(points[i], points[i - 1]);
            return ret;
        }
        /*面积*/
        FLOAT_ area()
        {
            FLOAT_ ret = Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + Vector2::Cross(points[i - 1], points[i]);
            return ret / 2;
        }
        /*求几何中心（形心、重心）*/
        Vector2 center()
        {
            Vector2 ret = (points.back() + points.front()) * Vector2::Cross(points.back(), points.front());
            for (int i = 1; i < points.size(); i++)
                ret = ret + (points[i - 1] + points[i]) * Vector2::Cross(points[i - 1], points[i]);
            return ret / area() / 6;
        }
        /*求边界整点数*/
        long long boundary_points()
        {
            long long b = 0;
            for (int i = 0; i < points.size() - 1; i++)
            {
                b += std::__gcd((long long)abs(points[i + 1].x - points[i].x), (long long)abs(points[i + 1].y - points[i].y));
            }
            return b;
        }
        /*Pick定理：多边形面积=内部整点数+边界上的整点数/2-1；求内部整点数*/
        long long interior_points(FLOAT_ A = -1, long long b = -1)
        {
            if (A < 0)
                A = area();
            if (b < 0)
                b = boundary_points();
            return (long long)A + 1 - (b / 2);
        }
    };

}