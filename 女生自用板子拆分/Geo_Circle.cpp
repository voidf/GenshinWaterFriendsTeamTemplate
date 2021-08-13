#include "Geo_Base.cpp"
#include "Geo_Vector2.cpp"
#include "Geo_Line2.cpp"

namespace Geometry
{
    /* https://www.luogu.com.cn/record/51674409 模板题需要用long double */
    struct Circle
    {
        Vector2 center;
        FLOAT_ radius;
        Circle(Vector2 c, FLOAT_ r) : center(c), radius(r) {}
        Circle(FLOAT_ x, FLOAT_ y, FLOAT_ r) : center(x, y), radius(r) {}
        Circle(Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 p1 = Vector2::LerpUnclamped(a, b, 0.5);
            Vector2 v1 = b - a;
            swap(v1.x, v1.y);
            v1.x = -v1.x;
            Vector2 p2 = Vector2::LerpUnclamped(b, c, 0.5);
            Vector2 v2 = c - b;
            swap(v2.x, v2.y);
            v2.x = -v2.x;

            center = Line2::Intersect(Line2(p1, v1, false), Line2(p2, v2, false));

            radius = (center - a).magnitude();
        }
        Vector2 fromRad(FLOAT_ A)
        {
            return Vector2(center.x + radius * cos(A), center.y + radius * sin(A));
        }
        std::pair<Vector2, Vector2> intersect_points(Line2 l)
        {
            FLOAT_ k = l.k();
            // 特判
            if (isnan(k))
            {
                FLOAT_ x = -l.C / l.A;
                FLOAT_ rhs = pow(radius, 2) - pow(x - center.x, 2);
                if (rhs < 0)
                    return make_pair(Vector2(nan(""), nan("")), Vector2(nan(""), nan("")));
                else
                {
                    rhs = sqrt(rhs);
                    return make_pair(Vector2(x, rhs + radius), Vector2(x, -rhs + radius));
                }
            }
            FLOAT_ lb = l.b();
            FLOAT_ a = k * k + 1;
            FLOAT_ b = 2 * k * (lb - center.y) - 2 * center.x;
            FLOAT_ c = pow(lb - center.y, 2) + pow(center.x, 2) - pow(radius, 2);
            FLOAT_ x1, x2;
            std::tie(x1, x2) = solveQuadraticEquation(a, b, c);
            if (isnan(x1))
            {
                return make_pair(Vector2(nan(""), nan("")), Vector2(nan(""), nan("")));
            }
            else
            {
                return make_pair(Vector2(x1, l.y(x1)), Vector2(x2, l.y(x2)));
            }
        }
        /* 使用极角和余弦定理算交点，更稳，但没添加处理相离和相包含的情况 */
        std::pair<Vector2, Vector2> intersect_points(Circle cir)
        {
            Vector2 distV = (cir.center - center);
            FLOAT_ dist = distV.magnitude();
            FLOAT_ ang = distV.toPolarAngle(false);
            FLOAT_ dang = acos((pow(radius, 2) + pow(dist, 2) - pow(cir.radius, 2)) / (2 * radius * dist)); //余弦定理
            return make_pair(fromRad(ang + dang), fromRad(ang - dang));
        }

        FLOAT_ area() { return PI * radius * radius; }

        bool is_outside(Vector2 p)
        {
            return (p - center).magnitude() > radius;
        }
        bool is_inside(Vector2 p)
        {
            return intereps((p - center).magnitude() - radius) < 0;
        }
        static intersect_area(Circle A, Circle B)
        {
            Vector2 dis = A.center - B.center;
            FLOAT_ sqrdis = dis.sqrMagnitude();
            FLOAT_ cdis = sqrt(sqrdis);
            if (sqrdis >= pow(A.radius + B.radius, 2))
                return FLOAT_(0);
            if (A.radius >= B.radius)
                std::swap(A, B);
            if (cdis + A.radius <= B.radius)
                return PI * A.radius * A.radius;
            if (sqrdis >= B.radius * B.radius)
            {
                FLOAT_ area = 0.0;
                FLOAT_ ed = sqrdis;
                FLOAT_ jiao = ((FLOAT_)B.radius * B.radius + ed - A.radius * A.radius) / (2.0 * B.radius * sqrt((FLOAT_)ed));
                jiao = acos(jiao);
                jiao *= 2.0;
                area += B.radius * B.radius * jiao / 2;
                jiao = sin(jiao);
                area -= B.radius * B.radius * jiao / 2;
                jiao = ((FLOAT_)A.radius * A.radius + ed - B.radius * B.radius) / (2.0 * A.radius * sqrt((FLOAT_)ed));
                jiao = acos(jiao);
                jiao *= 2;
                area += A.radius * A.radius * jiao / 2;
                jiao = sin(jiao);
                area -= A.radius * A.radius * jiao / 2;
                return area;
            }
            FLOAT_ area = 0.0;
            FLOAT_ ed = sqrdis;
            FLOAT_ jiao = ((FLOAT_)A.radius * A.radius + ed - B.radius * B.radius) / (2.0 * A.radius * sqrt(ed));
            jiao = acos(jiao);
            area += A.radius * A.radius * jiao;
            jiao = ((FLOAT_)B.radius * B.radius + ed - A.radius * A.radius) / (2.0 * B.radius * sqrt(ed));
            jiao = acos(jiao);
            area += B.radius * B.radius * jiao - B.radius * sqrt(ed) * sin(jiao);
            return area;
        }
    };

}