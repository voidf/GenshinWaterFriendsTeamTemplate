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

        /* 三角形面积并，只能处理三角形数组 */
		static FLOAT_ triangles_area(const std::vector<Polygon2> &P)
		{
			std::vector<FLOAT_> events;
			FLOAT_ ans = 0;
			for (int i = 0; i < P.size(); ++i)
			{
				for (int it = 0; it < 3; ++it)
				{
					// int ti = it == 0 ? 2 : it - 1;
					const Vector2 &ip1 = P[i].points[it];
					events.emplace_back(ip1.x);
					const Vector2 &ip2 = P[i].points[it ? it - 1 : 2];
					for (int j = i + 1; j < P.size(); ++j)
					{

						for (int jt = 0; jt < 3; ++jt)
						{
							const Vector2 &jp1 = P[j].points[jt];
							const Vector2 &jp2 = P[j].points[jt ? jt - 1 : 2];
							Segment2 si(ip1, ip2);
							Segment2 sj(jp1, jp2);
							if (Segment2::IsIntersect(si, sj) && !Segment2::IsParallel(si, sj))
							{
								events.emplace_back(Line2::Intersect(si, sj).x);
							}
						}
					}
				}
			}
			std::sort(events.begin(), events.end());
			events.resize(std::unique(events.begin(), events.end()) - events.begin());
			FLOAT_ bck = 0;
			std::map<FLOAT_, FLOAT_> M;
			FLOAT_ cur = 0;
			auto mergeseg = [](FLOAT_ l, FLOAT_ r, std::map<FLOAT_, FLOAT_> &M, FLOAT_ &cur) {
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
			for (int i = 0; i < events.size(); ++i)
			{
				cur = 0;
				FLOAT_ dx = i > 0 ? events[i] - events[i - 1] : 0;
				FLOAT_ cx = events[i];
				std::vector<std::pair<FLOAT_, FLOAT_>> leftborder, rightborder;
				M.clear();

				for (int j = 0; j < P.size(); ++j)
				{
					std::vector<FLOAT_> its;
					for (int jt = 0; jt < 3; ++jt)
					{
						const Vector2 &jp1 = P[j].points[jt];
						const Vector2 &jp2 = P[j].points[jt ? jt - 1 : 2];
						bool fg = 1;
						if (jp1.x == cx)
						{
							its.emplace_back(jp1.y);
							fg = 0;
						}
						if (jp2.x == cx)
						{
							its.emplace_back(jp2.y);
							fg = 0;
						}
						if (fg && ((jp1.x < cx) ^ (cx < jp2.x)) == 0)
						{
							Segment2 sj(jp1, jp2);
							its.emplace_back(sj.y(cx));
						}
					}
					if (its.size() <= 1)
						continue;
					char flg = 0;
					if (its.size() == 4)
					{
						flg = 'R';
						for (auto &p : P[j].points)
							if (p.x > cx)
							{
								flg = 'L';
								break;
							}
					}

					sort(its.begin(), its.end());
					if (flg == 'L')
					{
						leftborder.emplace_back(its.front(), its.back());
						continue;
					}
					if (flg == 'R')
					{
						rightborder.emplace_back(its.front(), its.back());
						continue;
					}
					mergeseg(its.front(), its.back(), M, cur);
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