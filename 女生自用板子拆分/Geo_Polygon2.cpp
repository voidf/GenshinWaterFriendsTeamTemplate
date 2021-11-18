#include "Geo_Base.cpp"
#include "Geo_Vector2.cpp"
#include "Geo_Segment2.cpp"

namespace Geometry
{

    struct Polygon2
    {
        std::vector<Vector2> points;

    private:
        Vector2 accordance;

    public:
        /* 求凸包 */
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

        /* log2(n)判断点在凸包内，要求逆时针序的凸包，即使用ConvexHull得到的多边形 */
        inline bool is_inner_convexhull(const Vector2 &p) const
        {
            int l = 1, r = points.size() - 2;
            while (l <= r)
            {
                int mid = l + r >> 1;
                FLOAT_ a1 = Vector2::Cross(points[mid] - points[0], p - points[0]);
                FLOAT_ a2 = Vector2::Cross(points[mid + 1] - points[0], p - points[0]);
                if (a1 >= 0 && a2 <= 0)
                {
                    if (Vector2::Cross(points[mid + 1] - points[mid], p - points[mid]) >= 0)
                        return 1;
                    return 0;
                }
                else if (a1 < 0)
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            return 0;
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
                avg += i;
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
                b += std::__gcd((long long)abs(points[i + 1].x - points[i].x), (long long)abs(points[i + 1].y - points[i].y));
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

        /* 别人写的更快的板子 */
        static FLOAT_ triangles_area(std::vector<Polygon2> &P)
        {
            int pos = 0;
            for (auto &i : P)
            {
                if (abs(Vector2::Cross(i.points[1] - i.points[0], i.points[2] - i.points[0])) < 1e-12)
                    continue;
                P[pos++] = i;
            }
            FLOAT_ ans = 0;
            for (int i = 0; i < P.size(); ++i)
                for (int j = 0; j < 3; ++j)
                {
                    std::vector<pair<FLOAT_, int>> ev({make_pair(0, 1), make_pair(1, -1)});
                    Vector2 s = P[i].points[j], t = P[i].points[(j + 1) % 3], r = P[i].points[(j + 2) % 3];
                    if (abs(s.x - t.x) <= 1e-12)
                        continue;
                    if (s.x > t.x)
                        swap(s, t);
                    int flag = Vector2::Cross(r - s, t - s) < 0 ? -1 : 1;
                    FLOAT_ stdis = (t - s).sqrMagnitude();
                    for (int i1 = 0; i1 < P.size(); ++i1)
                        if (i1 != i)
                        {
                            int pos[3] = {};
                            int cnt[3] = {};
                            for (int j1 = 0; j1 < 3; ++j1)
                            {
                                const Vector2 &p = P[i1].points[j1];
                                FLOAT_ area = Vector2::Cross(p - s, t - s);
                                if (area * area * 1e12 < stdis)
                                    pos[j1] = 0; // online
                                else
                                    pos[j1] = area > 0 ? 1 : -1;
                                ++cnt[pos[j1] + 1];
                            }
                            if (cnt[1] == 2)
                            {
                                FLOAT_ l = 1, r = 0;
                                int _j = -1;
                                for (int j1 = 0; j1 < 3; ++j1)
                                    if (pos[j1] == 0)
                                    {
                                        const Vector2 &p = P[i1].points[j1];
                                        FLOAT_ now = Vector2::Dot(p - s, t - s) / stdis;
                                        l = min(l, now);
                                        r = max(r, now);
                                        if (pos[(j1 + 1) % 3] == 0)
                                            _j = j1;
                                    }
                                Vector2 _s = P[i1].points[_j], _t = P[i1].points[(_j + 1) % 3], _r = P[i1].points[(_j + 2) % 3];
                                if (_s.x > _t.x)
                                    swap(_s, _t);
                                int _flag = Vector2::Cross(_r - _s, _t - _s) < 0 ? -1 : 1;
                                if (i1 > i && flag == _flag)
                                    continue;
                                l = max(l, 0.0);
                                r = min(r, 1.0);
                                if (l < r)
                                {
                                    ev.emplace_back(l, -1);
                                    ev.emplace_back(r, 1);
                                }
                                continue;
                            }
                            if (!cnt[0] || !cnt[2]) // 不过这条线
                                continue;
                            FLOAT_ l = 1, r = 0;
                            for (int j1 = 0; j1 < 3; ++j1)
                                if (pos[j1] == 0) // 在线上
                                {
                                    const Vector2 &p = P[i1].points[j1];
                                    FLOAT_ now = Vector2::Dot(p - s, t - s) / stdis;
                                    l = min(l, now);
                                    r = max(r, now);
                                }
                                else if (pos[j1] * pos[(j1 + 1) % 3] < 0) // 穿过
                                {
                                    Vector2 p0 = P[i1].points[j1], p1 = P[i1].points[(j1 + 1) % 3];
                                    FLOAT_ now = Vector2::Cross(p0 - s, p1 - p0) / Vector2::Cross(t - s, p1 - p0);
                                    l = min(l, now);
                                    r = max(r, now);
                                }
                            l = max(l, 0.0);
                            r = min(r, 1.0);
                            if (l < r)
                            {
                                ev.emplace_back(l, -1);
                                ev.emplace_back(r, 1);
                            }
                        }
                    sort(ev.begin(), ev.end());
                    FLOAT_ la = 0;
                    int sum = 0;
                    Vector2 a = t - s;
                    for (auto p : ev)
                    {
                        FLOAT_ t;
                        int v;
                        tie(t, v) = p;
                        if (sum > 0)
                            ans += flag * a.x * (t - la) * (s.y + a.y * (t + la) / 2);
                        sum += v;
                        la = t;
                    }
                }
            return ans;
        }
        /* 点光源在多边形上的照明段，点严格在多边形内，n^2极坐标扫描线 */
        std::vector<std::pair<Vector2, Vector2>> project_on_poly(const Vector2 &v)
        {
            std::vector<std::pair<Vector2, Vector2>> ret;
            int pvno = -1;
            Polygon2 p(*this);
            for (auto &i : p.points)
                i -= v;
            std::vector<Segment2> relative(1, Segment2(p.points.back(), p.points.front()));
            for (int i = 1; i < p.points.size(); ++i)
                relative.emplace_back(p.points[i - 1], p.points[i]);
            std::sort(p.points.begin(), p.points.end(), PolarSortCmp());

            for (int i = 0; i < p.points.size(); ++i) // x轴正向开始逆时针序
            {
                const Vector2 &p1 = p.points[i];
                const Vector2 &p2 = p.points[(i + 1) % p.points.size()];
                if (Vector2::Cross(p1, p2) == 0) // 共线，即使有投影，三角形也会退化成一条线，故忽略
                    continue;
                Vector2 mid = Vector2::SlerpUnclamped(p1, p2, 0.5);
                Segment2 midseg(0, mid);
                FLOAT_ nearest = -1;
                int sid = -1;
                for (int j = 0; j < relative.size(); ++j)
                    if (midseg.ray_in_range(relative[j]))
                    {
                        Vector2 its = Line2::Intersect(midseg, relative[j]);
                        if (Vector2::Dot(its, mid) > 0)
                        {
                            FLOAT_ d = its.sqrMagnitude();
                            if (nearest == -1 || nearest > d)
                            {
                                nearest = d;
                                sid = j;
                            }
                        }
                    }
                if (pvno == sid)
                    ret.back().second = v + Line2::Intersect(Line2(0, p2), relative[sid]);
                else
                {
                    pvno = sid;
                    ret.emplace_back(
                        v + Line2::Intersect(Line2(0, p1), relative[sid]),
                        v + Line2::Intersect(Line2(0, p2), relative[sid]));
                }
            }
            return ret;
        }

        /* 三角形面积并，只能处理三角形数组 */
        static FLOAT_ triangles_area_s(const std::vector<Polygon2> &P)
        {
            std::vector<FLOAT_> events;
            events.reserve(P.size() * P.size() * 9);
            FLOAT_ ans = 0;
            for (int i = 0; i < P.size(); ++i)
            {
                for (int it = 0; it < 3; ++it)
                {
                    const Vector2 &ip1 = P[i].points[it];
                    events.emplace_back(ip1.x);
                    const Vector2 &ip2 = P[i].points[it ? it - 1 : 2];
                    for (int j = i + 1; j < P.size(); ++j)

                        for (int jt = 0; jt < 3; ++jt)
                        {
                            const Vector2 &jp1 = P[j].points[jt];
                            const Vector2 &jp2 = P[j].points[jt ? jt - 1 : 2];
                            Segment2 si(ip1, ip2);
                            Segment2 sj(jp1, jp2);
                            if (Segment2::IsIntersect(si, sj) && !Segment2::IsParallel(si, sj))
                                events.emplace_back(Line2::Intersect(si, sj).x);
                        }
                }
            }
            std::sort(events.begin(), events.end());
            events.resize(std::unique(events.begin(), events.end()) - events.begin());
            FLOAT_ bck = 0;
            std::map<FLOAT_, FLOAT_> M;
            FLOAT_ cur = 0;
            auto mergeseg = [](FLOAT_ l, FLOAT_ r, std::map<FLOAT_, FLOAT_> &M, FLOAT_ &cur)
            {
                auto pos = M.upper_bound(r);

                if (pos == M.begin())
                    M[l] = r, cur += r - l;
                else
                    while (1)
                    {
                        auto tpos = pos;
                        --tpos;
                        if (tpos->first <= l && l <= tpos->second)
                        {
                            cur += max(r, tpos->second) - tpos->second;
                            tpos->second = max(r, tpos->second);
                            break;
                        }
                        else if (l <= tpos->first && tpos->first <= r)
                        {
                            r = max(r, tpos->second);
                            cur -= tpos->second - tpos->first;
                            M.erase(tpos);
                            if (pos != M.begin())
                                continue;
                        }
                        M[l] = r, cur += r - l;
                        break;
                    }
            };
            std::vector<std::pair<FLOAT_, FLOAT_>> leftborder, rightborder;
            leftborder.reserve(P.size() * P.size() * 9);
            rightborder.reserve(P.size() * P.size() * 9);
            for (int i = 0; i < events.size(); ++i)
            {
                leftborder.clear();
                rightborder.clear();
                cur = 0;
                FLOAT_ dx = i > 0 ? events[i] - events[i - 1] : 0;
                FLOAT_ cx = events[i];
                M.clear();

                for (int j = 0; j < P.size(); ++j)
                {
                    // std::vector<FLOAT_> its;
                    int itsctr = 0;
                    FLOAT_ lb = INFINITY;
                    FLOAT_ rb = -INFINITY;
                    // FLOAT_ rb = *std::max_element(its.begin(), its.end());
                    for (int jt = 0; jt < 3; ++jt)
                    {
                        const Vector2 &jp1 = P[j].points[jt];
                        const Vector2 &jp2 = P[j].points[jt ? jt - 1 : 2];
                        bool fg = 1;
                        if (jp1.x == cx)
                            ++itsctr, lb = min(lb, jp1.y), rb = max(rb, jp1.y), fg = 0;
                        if (jp2.x == cx)
                            ++itsctr, lb = min(lb, jp2.y), rb = max(rb, jp2.y), fg = 0;
                        if (fg && ((jp1.x < cx) ^ (cx < jp2.x)) == 0)
                        {
                            Segment2 sj(jp1, jp2);
                            FLOAT_ cxy = sj.y(cx);
                            ++itsctr, lb = min(lb, cxy), rb = max(rb, cxy);
                        }
                    }
                    if (itsctr <= 1)
                        continue;
                    char flg = 0;
                    if (itsctr == 4)
                    {
                        flg = 'R';
                        for (auto &p : P[j].points)
                            if (p.x > cx)
                            {
                                flg = 'L';
                                break;
                            }
                    }

                    if (flg == 'L')
                    {
                        leftborder.emplace_back(lb, rb);
                        continue;
                    }
                    if (flg == 'R')
                    {
                        rightborder.emplace_back(lb, rb);
                        continue;
                    }
                    mergeseg(lb, rb, M, cur);
                }
                auto mcp = M;
                auto ccur = cur;
                while (rightborder.size())
                {
                    mergeseg(rightborder.back().first, rightborder.back().second, mcp, ccur);
                    rightborder.pop_back();
                }

                ans += i > 0 ? (ccur + bck) * dx : 0;
                while (leftborder.size())
                {
                    mergeseg(leftborder.back().first, leftborder.back().second, M, cur);
                    leftborder.pop_back();
                }
                bck = cur;
            }
            return ans * 0.5;
        }
    };
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
        ret.points.reserve(subdivision * 2);
        FLOAT_ step = 2 * PI / subdivision, cur = 0;
        while (subdivision--)
        {
            ret.points.emplace_back(center + Vector2::fromPolarCoordinate(Vector2(outer_radius, cur), false));
            ret.points.emplace_back(center + Vector2::fromPolarCoordinate(Vector2(inner_radius, cur + step / 2), false));
            cur += step;
        }
        return ret;
    }
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