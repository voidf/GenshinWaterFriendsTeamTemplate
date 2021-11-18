#include "Geo_Base.cpp"
#include "Geo_Vector2.cpp"

namespace Geometry
{

    struct Line2
    {
        FLOAT_ A, B, C;
        /* 默认两点式，打false为点向式（先点后向） */
        Line2(const Vector2 &u, const Vector2 &v, bool two_point = true) : A(u.y - v.y), B(v.x - u.x), C(u.y * (u.x - v.x) - u.x * (u.y - v.y))
        {
            if (u == v)
            {
                if (u.x)
                {
                    A = 1;
                    B = 0;
                    C = -u.x;
                }
                else if (u.y)
                {
                    A = 0;
                    B = 1;
                    C = -u.y;
                }
                else
                {
                    A = 1;
                    B = -1;
                    C = 0;
                }
            }
            if (!two_point)
            {
                A = -v.y;
                B = v.x;
                C = -(A * u.x + B * u.y);
            }
        }
        Line2(FLOAT_ a, FLOAT_ b, FLOAT_ c) : A(a), B(b), C(c) {}
        std::string ToString() const
        {
            std::ostringstream ostr;
            ostr << "Line2(" << this->A << ", " << this->B << ", " << this->C << ")";
            return ostr.str();
        }
        friend std::ostream &operator<<(std::ostream &o, const Line2 &v)
        {
            o << v.ToString();
            return o;
        }
        static FLOAT_ getk(Vector2 &u, Vector2 &v) { return (v.y - u.y) / (v.x - u.x); }
        FLOAT_ k() const { return -A / B; }
        FLOAT_ b() const { return -C / B; }
        FLOAT_ x(FLOAT_ y) const { return -(B * y + C) / A; }
        FLOAT_ y(FLOAT_ x) const { return -(A * x + C) / B; }
        /* 点到直线的距离 */
        FLOAT_ distToPoint(const Vector2 &p) const { return abs(A * p.x + B * p.y + C / sqrt(A * A + B * B)); }
        /* 直线距离公式，使用前先判平行 */
        static FLOAT_ Distance(const Line2 &a, const Line2 &b) { return abs(a.C - b.C) / sqrt(a.A * a.A + a.B * a.B); }
        /* 判断平行 */
        static bool IsParallel(const Line2 &u, const Line2 &v)
        {
            bool f1 = round_compare(u.B, 0.0);
            bool f2 = round_compare(v.B, 0.0);
            if (f1 != f2)
                return false;
            return f1 or round_compare(u.A * v.B - v.A * u.B, 0);
        }

        /* 单位化（？） */
        void normalize()
        {
            FLOAT_ su = sqrt(A * A + B * B + C * C);
            if (A < 0)
                su = -su;
            else if (A == 0 and B < 0)
                su = -su;
            A /= su;
            B /= su;
            C /= su;
        }
        /* 返回单位化后的直线 */
        Line2 normalized() const
        {
            Line2 t(*this);
            t.normalize();
            return t;
        }

        bool operator==(const Line2 &v) const { return round_compare(A, v.A) and round_compare(B, v.B) and round_compare(C, v.C); }
        bool operator!=(const Line2 &v) const { return !(*this == v); }

        /* 判断两直线是否是同一条直线 */
        static bool IsSame(const Line2 &u, const Line2 &v)
        {
            return Line2::IsParallel(u, v) and round_compare(Distance(u.normalized(), v.normalized()), 0.0);
        }

        /* 计算交点 */
        static Vector2 Intersect(const Line2 &u, const Line2 &v)
        {
            FLOAT_ tx = (u.B * v.C - v.B * u.C) / (v.B * u.A - u.B * v.A);
            FLOAT_ ty = (u.B != 0.0 ? (-u.A * tx - u.C) / u.B : (-v.A * tx - v.C) / v.B);
            return Vector2(tx, ty);
        }
    };

}