#ifndef Geo_Face3_H
#define Geo_Face3_H

#include "Geo_Base.cpp"
#include "Geo_Vector3.cpp"
#include "Geo_Face3.cpp"

namespace Geometry
{
    /* 两点式空间直线，1 to 0 from */
	struct Segment3 : std::array<Vector3, 2>
	{
		Segment3(const Vector3 &v0, const Vector3 &v1) : std::array<Vector3, 2>({v0, v1}) {}
		template <typename... Args>
		Segment3(bool super, Args &&...args) : std::array<Vector3, 2>(std::forward<Args>(args)...) {}
		/* 方向向量，未经单位化 */
		Vector3 toward() const { return at(1) - at(0); }
		/* 点到空间直线的距离，一次sqrt */
		FLOAT_ distance(const Vector3 &p) const
		{
			Vector3 p1 = toward();
			Vector3 p2 = p - at(0);
			Vector3 c = Vector3::Cross(p1, p2);
			return sqrt(c.sqrMagnitude() / p1.sqrMagnitude()); // 损失精度的源泉：sqrt
		}
		/* 点到空间直线的垂足，无精度损失 */
		Vector3 project(const Vector3 &p) const
		{
			Vector3 p1 = toward();
			Vector3 p2 = p - at(0);
			// cerr << cos(Vector3::Rad(p2, p1)) << endl;
			// cerr << p1.normalized() << endl;
			// FLOAT_ r = Vector3::Rad(p2, p1);
			// Vector3 c = Vector3::Cross(p1, p2);
			// c.len / p1.len * p1 / p1.len
			// return at(0) + Vector3::Project(p2, p1);
			return Vector3::Dot(p2, p1) * p1 / p1.sqrMagnitude() + at(0); // 无损的式子化简
																		  // return Vector3::Cos(p2, p1) * p1 * sqrt(p2.sqrMagnitude() / p1.sqrMagnitude()) + at(0); // 损失精度源：
		}
		/* 直线与平面交点，无损 */
		Vector3 intersect(const Face3 &f) const
		{
			// FLOAT_ a0 = f.distanceS(at(0));
			// FLOAT_ a1 = f.distanceS(at(1));
			FLOAT_ a00 = Vector3::Dot(at(0) - f.at(0), f.normal());
			FLOAT_ a11 = Vector3::Dot(at(1) - f.at(0), f.normal());
			// Vector3 d0 = a0 * toward() / (a0 - a1); // 两个sqrt
			Vector3 d0 = a00 * toward() / (a00 - a11); // 无损

			return d0 + at(0);
		}
		/* 异面直线最近点对，无损 */
		std::pair<Vector3, Vector3> nearest(const Segment3 &s) const
		{
			Vector3 p1 = toward();
			Vector3 p2 = s.at(0) - at(0);
			Vector3 p3 = s.at(1) - at(0);

			Vector3 c = Vector3::Cross(p1, s.toward());
			Face3 f(at(0), c + at(0), p1 + at(0));

			Vector3 sret = s.intersect(f);
			Vector3 pj = project(sret);
			return std::make_pair(isnan(pj.x) ? sret : pj, sret);
		}
		/* 空间直线的距离，一次sqrt */
		FLOAT_ distance(const Segment3 &s) const
		{
			if (Vector3::coplanar({at(1), at(0), s.at(1), s.at(0)}))
				return distance(s.at(0));
			Vector3 c = Vector3::Cross(toward(), s.toward());
			c.Normalize();
			return abs(Vector3::Dot(c, at(0) - s.at(0)));
			// auto sol = nearest(s);
			// return Vector3::Distance(sol.first, sol.second);
		}
	};
    
}

#endif