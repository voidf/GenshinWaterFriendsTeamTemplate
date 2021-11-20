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

/* 旋转坐标系求最小三角形

int n, m, T, tot, q, s, t, k, L, R, x, y;

short rank_[5000], rank2index[5000];
int prank = 0, prank2index = 0;
using lineinfo = tuple<short, short>;

int plineinfo = 0;

void solve()
{
    qr(n);
    using namespace Geometry;
    vector<Vector2> points;
    points.reserve(n);

    vector<lineinfo> lines;
    lines.reserve(n * (n - 1) >> 1);

    for (auto i : range(n))
    {
        long long x, y;
        qr(x);
        qr(y);
        points.emplace_back(x, y);
    }
    FLOAT_ ans = 0x3f3f3f3f3f3f3f3f;

    sort(points.begin(), points.end());
    for (auto i : range(n))
    {
        rank_[i] = i;
        rank2index[i] = i;
        for (auto j : range(i + 1, n))
        {
            // lines.emplace_back(i, j, Line2(points[i], points[j]).k());
            lines.emplace_back(i, j);
        }
    }

    sort(
        lines.begin(), lines.end(), [&](const lineinfo &a, const lineinfo &b) -> bool
        {
// (get<0>(a).y - get<1>(a).y)*(get<0>(a).y - get<1>(a).y)
#define Au points[get<0>(a)]
#define Bu points[get<0>(b)]
#define Av points[get<1>(a)]
#define Bv points[get<1>(b)]
            return (Av.y - Au.y) * (Bv.x - Bu.x) < (Bv.y - Bu.y) * (Av.x - Au.x);
            // auto k1 = Line2::getk(points[get<0>(a)], points[get<1>(a)]);
            // auto k2 = Line2::getk(points[get<0>(b)], points[get<1>(b)]);
            // return k1 < k2 || k1 == k2 && a < b;
        });

    for (auto i : range(lines.size()))
    {
        short A, B;
        tie(A, B) = lines[i];
        short rka = rank_[A], rkb = rank_[B];
        if (rka > rkb)
            swap(rka, rkb);
        if (rka > 0)
            ans = min(ans, abs(Vector2::Cross(points[A] - points[B], points[rank2index[rka - 1]] - points[B])));
        if (rkb < n - 1)
            ans = min(ans, abs(Vector2::Cross(points[A] - points[B], points[rank2index[rkb + 1]] - points[B])));

        if (ans == 0.0)
            break;
        swap(rank2index[rka], rank2index[rkb]);
        swap(rank_[A], rank_[B]);
    }

    printf("%.3lf\n", (double)ans * 0.5);
}

*/