#ifndef Geo_Face3_H
#define Geo_Face3_H

#include "Geo_Base.cpp"
#include "Geo_Vector3.cpp"

namespace Geometry
{
	struct Face3 : std::array<Vector3, 3>
	{
		inline void constructCoordinate(const Vector3 &z, Vector3 &x, Vector3 &y)
		{
			if (std::abs(z.x) > std::abs(z.y))
				y = Vector3(z.z, 0, -z.x);
			else
				y = Vector3(0, z, z, -z, y);
			x = Vector3::Cross(y, z);
		}
		Face3(const Vector3 &normal, const Vector3 &offset)
		{
			Vector3 x, y;
			constructCoordinate(normal, x, y);
			(*this)[0] = offset;
			(*this)[1] = offset + x;
			(*this)[2] = offset + y;
		}
		Face3(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) : std::array<Vector3, 3>({v0, v1, v2}) {}
		template <typename... Args>
		Face3(bool super, Args &&...args) : std::array<Vector3, 3>(std::forward<Args>(args)...) {}
		inline static Vector3 normal(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) { return Vector3::Cross(v1 - v0, v2 - v0); }
		inline static FLOAT_ area(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) { return normal(v0, v1, v2).magnitude() / FLOAT_(2); }
		inline static bool visible(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, const Vector3 &_v) { return Vector3::Dot(_v - v0, normal(v0, v1, v2)) > 0; }
		/* 未经单位化的法向 */
		inline Vector3 normal() const { return Vector3::Cross(at(1) - at(0), at(2) - at(0)); }
		inline FLOAT_ area() const { return normal().magnitude() / FLOAT_(2); }
		inline bool visible(const Vector3 &_v) const { return Vector3::Dot(_v - at(0), normal()) > 0; }
		/* 点到平面代数距离，一次sqrt */
		inline FLOAT_ distanceS(const Vector3 &p) const { return Vector3::Dot(p - at(0), normal().normalized()); }
		/* 点到平面的投影，无损 */
		inline Vector3 project(const Vector3 &p) const { return p - normal() * Vector3::Dot(p - at(0), normal()) / normal().sqrMagnitude(); }
	};
}

#endif