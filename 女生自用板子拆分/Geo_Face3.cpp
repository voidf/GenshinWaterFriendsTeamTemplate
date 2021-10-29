#ifndef Geo_Face3_H
#define Geo_Face3_H

#include "Geo_Base.cpp"
#include "Geo_Vector3.cpp"

namespace Geometry
{
    struct Face3 : std::array<Vector3, 3>
    {
        Face3(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) : std::array<Vector3, 3>({v0, v1, v2}) {}
        inline static Vector3 normal(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) { return Vector3::Cross(v1 - v0, v2 - v0); }
        inline static FLOAT_ area(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) { return normal(v0, v1, v2).magnitude() / FLOAT_(2); }
        inline static bool visible(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, const Vector3 &_v) { return Vector3::Dot(_v - v0, normal(v0, v1, v2)) > 0; }
        inline Vector3 normal() { return Vector3::Cross((*this)[1] - (*this)[0], (*this)[2] - (*this)[0]); }
        inline FLOAT_ area() { return normal().magnitude() / FLOAT_(2); }
        inline bool visible(const Vector3 &_v) { return Vector3::Dot(_v - (*this)[0], normal()) > 0; }
    };
}

#endif