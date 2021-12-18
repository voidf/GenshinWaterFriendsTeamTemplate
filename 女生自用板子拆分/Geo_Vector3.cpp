#ifndef Geo_vec3_H
#define Geo_vec3_H

#include "Geo_Base.cpp"

namespace Geometry
{
    struct vec3 // 三维向量
    {
        fl x, y, z;
        vec3(fl _x, fl _y, fl _z) : x(_x), y(_y), z(_z) {}
        vec3(fl n) : x(n), y(n), z(n) {}
        vec3() : x(0.0), y(0.0), z(0.0) {}
        inline vec3 &operator=(const vec3 &b)
        {
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            return *this;
        }
        inline bool operator==(const vec3 &b) const { return round_compare(this->x, b.x) and round_compare(this->y, b.y) and round_compare(this->z, b.z); }
        inline bool operator!=(const vec3 &b) const { return not((*this) == b); }
        inline fl &operator[](const int ind)
        {
            switch (ind)
            {
            case 0:
                return this->x;
                break;
            case 1:
                return this->y;
                break;
            case 2:
                return this->z;
                break;
            case 'x':
                return this->x;
                break;
            case 'y':
                return this->y;
                break;
            case 'z':
                return this->z;
            default:
                throw "无法理解除0,1,2外的索引";
                break;
            }
        }
        inline friend std::ostream &operator<<(std::ostream &o, const vec3 &v) { return o << v.ToString(); }
        inline vec3 &operator+=(const vec3 &b)
        {
            x += b.x, y += b.y, z += b.z;
            return (*this);
        }
        inline vec3 &operator-=(const vec3 &b)
        {
            x -= b.x, y -= b.y, z -= b.z;
            return (*this);
        }
        inline vec3 &operator*=(const vec3 &b)
        {
            x *= b.x, y *= b.y, z *= b.z;
            return (*this);
        }
        inline vec3 &operator/=(const vec3 &b)
        {
            x /= b.x, y /= b.y, z /= b.z;
            return (*this);
        }
        inline vec3 &operator+=(const fl &n)
        {
            x += n, y += n, z += n;
            return (*this);
        }
        inline vec3 &operator-=(const fl &n)
        {
            x -= n, y -= n, z -= n;
            return (*this);
        }
        inline vec3 &operator*=(const fl &n)
        {
            x *= n, y *= n, z *= n;
            return (*this);
        }
        inline vec3 &operator/=(const fl &n)
        {
            x /= n, y /= n, z /= n;
            return (*this);
        }
        inline vec3 operator+(const vec3 &b) const { return vec3(*this) += b; }
        inline vec3 operator-(const vec3 &b) const { return vec3(*this) -= b; }
        inline vec3 operator*(const vec3 &b) const { return vec3(*this) *= b; }
        inline vec3 operator/(const vec3 &b) const { return vec3(*this) /= b; }
        inline vec3 operator+(const fl &n) const { return vec3(*this) += n; }
        inline vec3 operator-(const fl &n) const { return vec3(*this) -= n; }
        inline vec3 operator*(const fl &n) const { return vec3(*this) *= n; }
        inline vec3 operator/(const fl &n) const { return vec3(*this) /= n; }
        inline friend vec3 operator+(const fl &n, const vec3 &b) { return vec3(n) += b; }
        inline friend vec3 operator-(const fl &n, const vec3 &b) { return vec3(n) -= b; }
        inline friend vec3 operator*(const fl &n, const vec3 &b) { return vec3(n) *= b; }
        inline friend vec3 operator/(const fl &n, const vec3 &b) { return vec3(n) /= b; }

        /* 向量的平方模 */
        inline fl sqrMagnitude() const { return x * x + y * y + z * z; }
        /* 向量的模，一次sqrt */
        inline fl magnitude() const { return sqrt(this->sqrMagnitude()); }
        /* 判等 */
        inline bool equals(const vec3 &b) const { return (*this) == b; }
        /* 向量单位化，一次sqrt */
        inline void Normalize()
        {
            fl _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
            this->z /= _m;
        }

        /* 转为字符串 */
        inline std::string ToString() const
        {
            std::ostringstream ostr;
            ostr << "vec3(" << this->x << ", " << this->y << ", " << this->z << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量，一次sqrt */
        inline vec3 normalized() const
        {
            fl _m = this->magnitude();
            return vec3(this->x / _m, this->y / _m, this->z / _m);
        }
        /* 距离，一次sqrt */
        inline static fl Distance(const vec3 &a, const vec3 &b) { return (a - b).magnitude(); }

        /* 向量线性插值 */
        inline static vec3 lerp(const vec3 &a, const vec3 &b, const fl &t) { return a + (b - a) * t; }

        /* 拿它的垂直向量（逆时针旋转90°） */
        inline static vec3 Perpendicular(const vec3 &inDirection) { return vec3(-inDirection.y, inDirection.x, 0); }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        inline static vec3 Reflect(const vec3 &inDirection, const vec3 &inNormal) { return inDirection - 2 * vec3::Dot(inDirection, inNormal) * inNormal; }

        /* 点积 */
        inline static fl Dot(const vec3 &lhs, const vec3 &rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
        /* 叉积 */
        inline static vec3 Cross(const vec3 &lhs, const vec3 &rhs) { return vec3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }

        /* 无符号夹角cos值，一次sqrt */
        inline static fl Cos(const vec3 &from, const vec3 &to) { return Dot(from, to) / sqrt(from.sqrMagnitude() * to.sqrMagnitude()); }
        /* 无符号弧度夹角，一次sqrt，一次acos */
        inline static fl Rad(const vec3 &from, const vec3 &to) { return acos(Cos(from, to)); }

        /* 无符号角度夹角，一次sqrt，一次acos，一次/PI  */
        inline static fl Angle(const vec3 &from, const vec3 &to) { return Rad(from, to) * 180 / PI; }

        /* 返回该方向上最大不超过maxLength长度的向量 */
        inline static vec3 ClampMagnitude(const vec3 &vector, const fl &maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /* 返回俩向量中x的最大值和y的最大值构造而成的向量 */
        inline static vec3 Max(const vec3 &lhs, const vec3 &rhs) { return vec3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)); }

        /* 返回俩向量中x的最小值和y的最小值构造而成的向量 */
        inline static vec3 Min(const vec3 &lhs, const vec3 &rhs) { return vec3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)); }

        /* 获得vector在onNormal方向的投影，无损，无需单位化写法 */
        inline static vec3 Project(const vec3 &vector, const vec3 &onNormal) { return Dot(vector, onNormal) / onNormal.sqrMagnitude() * onNormal; }

        /* 正交化：将两个向量单位化，并调整切线位置使之垂直于法向 */
        inline static void OrthoNormalize(vec3 &normal, vec3 &tangent)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
        }

        /* 正交化：将三个向量单位化，并调整使之两两垂直 */
        inline static void OrthoNormalize(vec3 &normal, vec3 &tangent, vec3 &binormal)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
            binormal -= Project(binormal, normal);
            binormal -= Project(binormal, tangent);
            binormal.Normalize();
        }

        /* 获得vector在以planeNormal为法向量的平面的投影，3个sqrt带一个sin，建议用Face3的project */
        inline static vec3 ProjectOnPlane(vec3 vector, vec3 planeNormal)
        {
            fl mag = vector.magnitude();
            fl s = Rad(vector, planeNormal);
            OrthoNormalize(planeNormal, vector);
            return mag * sin(s) * vector;
        }

        /* 罗德里格旋转公式，获得current绕轴normal(请自己单位化)旋转degree度（默认角度）的向量，右手螺旋意义，一个sin一个sqrt（算上normal单位化） */
        inline static vec3 Rotate(const vec3 &current, const vec3 &normal, const fl &degree, bool use_degree = 1)
        {
            fl r = use_degree ? degree / 180 * PI : degree;
            fl c = cos(r);
            return c * current + (1.0 - c) * Dot(normal, current) * normal + Cross(sin(r) * normal, current);
        }

        /* 将current向target转向degree度，如果大于夹角则返回target方向长度为current的向量 */
        inline static vec3 RotateTo(const vec3 &current, const vec3 &target, const fl &degree, bool use_degree = 1)
        {
            fl r = use_degree ? degree / 180 * PI : degree;
            if (r >= Rad(current, target))
                return current.magnitude() / target.magnitude() * target;
            else
            {
                // fl mag = current.magnitude();
                vec3 nm = Cross(current, target).normalized();
                return Rotate(current, nm, r);
            }
        }

        /* 球面插值 */
        inline static vec3 SlerpUnclamped(const vec3 &a, const vec3 &b, const fl &t)
        {
            vec3 rot = RotateTo(a, b, Rad(a, b) * t, false);
            fl l = b.magnitude() * t + a.magnitude() * (1 - t);
            return rot.normalized() * l;
        }

        /* 根据经纬，拿一个单位化的三维向量，以北纬和东经为正 */
        inline static vec3 FromLongitudeAndLatitude(const fl &longitude, const fl &latitude)
        {
            vec3 lat = Rotate(vec3(1, 0, 0), vec3(0, -1, 0), latitude);
            return Rotate(lat, vec3(0, 0, 1), longitude);
        }

        /* 球坐标转换为xyz型三维向量 */
        inline static vec3 FromSphericalCoordinate(const vec3 &spherical, bool use_degree = 1) { return FromSphericalCoordinate(spherical.x, spherical.y, spherical.z, use_degree); }
        /* 球坐标转换为xyz型三维向量，半径r，theta倾斜角（纬度），phi方位角（经度），默认输出角度 */
        inline static vec3 FromSphericalCoordinate(const fl &r, fl theta, fl phi, bool use_degree = 1)
        {
            theta = use_degree ? theta / 180 * PI : theta;
            phi = use_degree ? phi / 180 * PI : phi;
            return vec3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta));
        }
        /* 直角坐标转换为球坐标，默认输出角度 */
        inline static vec3 ToSphericalCoordinate(const vec3 &coordinate, bool use_degree = 1)
        {
            fl r = coordinate.magnitude();
            return vec3(
                r,
                acos(coordinate.z / r) * (use_degree ? 180.0 / PI : 1),
                atan2(coordinate.y, coordinate.x) * (use_degree ? 180.0 / PI : 1));
        }
        /* 直角坐标转换为球坐标，默认输出角度 */
        inline vec3 toSphericalCoordinate(bool use_degree = 1) { return ToSphericalCoordinate(*this, use_degree); }

        /* 判断四点共面 */
        static bool coplanar(const std::array<vec3, 4> &v)
        {
            vec3 v1 = v.at(1) - v.at(0);
            vec3 v2 = v.at(2) - v.at(0);
            vec3 v3 = v.at(3) - v.at(0);
            return vec3::Cross(vec3::Cross(v3, v1), vec3::Cross(v3, v2)).sqrMagnitude() == 0;
        }

        /* 判断三点共线 */
        static bool collinear(const std::array<vec3, 3> &v)
        {
            vec3 v1 = v.at(1) - v.at(0);
            vec3 v2 = v.at(2) - v.at(0);
            return vec3::Cross(v2, v1).sqrMagnitude() == 0;
        }
    };

}

#endif