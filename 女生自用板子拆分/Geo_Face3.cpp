#ifndef Geo_Face3_H
#define Geo_Face3_H

#include "Geo_Base.cpp"
#include "Geo_vec3.cpp"

namespace Geometry
{

	struct Face3 : std::array<vec3, 3>
	{
		/* 根据法向构造坐标系向量 */
		inline void constructCoordinate(const vec3 &z, vec3 &x, vec3 &y)
		{
			if (std::abs(z.x) > std::abs(z.y))
				y = vec3(z.z, 0, -z.x);
			else
				y = vec3(0, z.z, -z.y);
			x = vec3::Cross(y, z);
		}
		Face3(const vec3 &normal, const vec3 &offset)
		{
			vec3 x, y;
			constructCoordinate(normal, x, y);
			(*this)[0] = offset;
			(*this)[1] = offset + x;
			(*this)[2] = offset + y;
		}
		Face3(const vec3 &v0, const vec3 &v1, const vec3 &v2) : std::array<vec3, 3>({v0, v1, v2}) {}
		template <typename... Args>
		Face3(bool super, Args &&...args) : std::array<vec3, 3>(std::forward<Args>(args)...) {}
		inline static vec3 normal(const vec3 &v0, const vec3 &v1, const vec3 &v2) { return vec3::Cross(v1 - v0, v2 - v0); }
		inline static fl area(const vec3 &v0, const vec3 &v1, const vec3 &v2) { return normal(v0, v1, v2).magnitude() / fl(2); }
		inline static bool visible(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &_v) { return vec3::Dot(_v - v0, normal(v0, v1, v2)) > 0; }
		/* 未经单位化的法向 */
		inline vec3 normal() const { return vec3::Cross(at(1) - at(0), at(2) - at(0)); }
		inline fl area() const { return normal().magnitude() / fl(2); }
		inline bool visible(const vec3 &_v) const { return vec3::Dot(_v - at(0), normal()) > 0; }
		/* 点到平面代数距离，一次sqrt */
		inline fl distanceS(const vec3 &p) const { return vec3::Dot(p - at(0), normal().normalized()); }
		/* 点到平面的投影，无损 */
		inline vec3 project(const vec3 &p) const { return p - normal() * vec3::Dot(p - at(0), normal()) / normal().sqrMagnitude(); }
	};
}

#endif