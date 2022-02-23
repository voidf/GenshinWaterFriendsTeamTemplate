#ifndef Geo_Circle_H
#define Geo_Circle_H


#include "Geo_Base.cpp"
#include "Geo_vec2.cpp"
#include "Geo_Line2.cpp"

namespace Geometry
{
    /* https://www.luogu.com.cn/record/51674409 模板题需要用long double */
    struct Circle
    {
        vec2 center;
        fl radius;
        Circle(vec2 c, fl r) : center(c), radius(r) {}
        Circle(fl x, fl y, fl r) : center(x, y), radius(r) {}
        Circle(const vec2 &a, const vec2 &b, const vec2 &c)
        {
            vec2 p1 = vec2::lerp(a, b, 0.5);
            vec2 v1 = b - a;
            swap(v1.x, v1.y);
            v1.x = -v1.x;
            vec2 p2 = vec2::lerp(b, c, 0.5);
            vec2 v2 = c - b;
            swap(v2.x, v2.y);
            v2.x = -v2.x;

            center = Line2::Intersect(Line2(p1, v1, false), Line2(p2, v2, false));

            radius = (center - a).magnitude();
        }
        /* 圆上极角点转换为直角坐标 */
        vec2 fromRad(fl A) const
        {
            return vec2(center.x + radius * cos(A), center.y + radius * sin(A));
        }
        /* 圆与直线求交点 */
        std::pair<vec2, vec2> intersect_points(const Line2 &l) const
        {
            fl k = l.k();
            // 特判
            if (isnan(k))
            {
                fl x = -l.C / l.A;
                fl rhs = pow(radius, 2) - pow(x - center.x, 2);
                if (rhs < 0)
                    return make_pair(vec2(nan(""), nan("")), vec2(nan(""), nan("")));
                else
                {
                    rhs = sqrt(rhs);
                    return make_pair(vec2(x, rhs + radius), vec2(x, -rhs + radius));
                }
            }
            fl lb = l.b();
            fl a = k * k + 1;
            fl b = 2 * k * (lb - center.y) - 2 * center.x;
            fl c = pow(lb - center.y, 2) + pow(center.x, 2) - pow(radius, 2);
            fl x1, x2;
            std::tie(x1, x2) = solveQuadraticEquation(a, b, c);
            if (isnan(x1))
            {
                return make_pair(vec2(nan(""), nan("")), vec2(nan(""), nan("")));
            }
            else
            {
                return make_pair(vec2(x1, l.y(x1)), vec2(x2, l.y(x2)));
            }
        }
        /* 使用极角和余弦定理算交点，更稳，但没添加处理相离和相包含的情况 */
        std::pair<vec2, vec2> intersect_points(const Circle &cir) const
        {
            vec2 distV = (cir.center - center);
            fl dist = distV.magnitude();
            fl ang = distV.toPolarAngle(false);
            fl dang = acos((pow(radius, 2) + pow(dist, 2) - pow(cir.radius, 2)) / (2 * radius * dist)); //余弦定理
            return make_pair(fromRad(ang + dang), fromRad(ang - dang));
        }

        fl area() const { return PI * radius * radius; }

        bool is_outside(const vec2 &p) const
        {
            return (p - center).magnitude() > radius;
        }
        bool is_inside(const vec2 &p) const
        {
            return intereps((p - center).magnitude() - radius) < 0;
        }
        static intersect_area(const Circle &A, const Circle &B)
        {
            vec2 dis = A.center - B.center;
            fl sqrdis = dis.sqrMagnitude();
            fl cdis = sqrt(sqrdis);
            if (sqrdis >= pow(A.radius + B.radius, 2))
                return fl(0);
            if (A.radius >= B.radius)
                std::swap(A, B);
            if (cdis + A.radius <= B.radius)
                return PI * A.radius * A.radius;
            if (sqrdis >= B.radius * B.radius)
            {
                fl area = 0.0;
                fl ed = sqrdis;
                fl jiao = ((fl)B.radius * B.radius + ed - A.radius * A.radius) / (2.0 * B.radius * sqrt((fl)ed));
                jiao = acos(jiao);
                jiao *= 2.0;
                area += B.radius * B.radius * jiao / 2;
                jiao = sin(jiao);
                area -= B.radius * B.radius * jiao / 2;
                jiao = ((fl)A.radius * A.radius + ed - B.radius * B.radius) / (2.0 * A.radius * sqrt((fl)ed));
                jiao = acos(jiao);
                jiao *= 2;
                area += A.radius * A.radius * jiao / 2;
                jiao = sin(jiao);
                area -= A.radius * A.radius * jiao / 2;
                return area;
            }
            fl area = 0.0;
            fl ed = sqrdis;
            fl jiao = ((fl)A.radius * A.radius + ed - B.radius * B.radius) / (2.0 * A.radius * sqrt(ed));
            jiao = acos(jiao);
            area += A.radius * A.radius * jiao;
            jiao = ((fl)B.radius * B.radius + ed - A.radius * A.radius) / (2.0 * B.radius * sqrt(ed));
            jiao = acos(jiao);
            area += B.radius * B.radius * jiao - B.radius * sqrt(ed) * sin(jiao);
            return area;
        }
    };

}
#endif
