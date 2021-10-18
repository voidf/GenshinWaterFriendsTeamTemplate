#include "Geo_Base.cpp"

namespace Geometry
{

    struct Vector3 // 三维向量
    {
        FLOAT_ x, y, z;
        Vector3(FLOAT_ _x, FLOAT_ _y, FLOAT_ _z) : x(_x), y(_y), z(_z) {}
        Vector3(FLOAT_ n) : x(n), y(n), z(n) {}
        Vector3(const Vector2 &b) : x(b.x), y(b.y), z(0.0) {}
        Vector3() : x(0.0), y(0.0), z(0.0) {}
        inline Vector3 &operator=(const Vector3 &b)
        {
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            return *this;
        }
        inline bool operator==(const Vector3 &b) const { return round_compare(this->x, b.x) and round_compare(this->y, b.y) and round_compare(this->z, b.z); }
        inline bool operator!=(const Vector3 &b) const { return not((*this) == b); }
        inline FLOAT_ &operator[](const int ind)
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
        inline friend std::ostream &operator<<(std::ostream &o, const Vector3 &v) { return o << v.ToString(); }
        inline Vector3 &operator+=(const Vector3 &b)
        {
            x += b.x, y += b.y, z += b.z;
            return (*this);
        }
        inline Vector3 &operator-=(const Vector3 &b)
        {
            x -= b.x, y -= b.y, z -= b.z;
            return (*this);
        }
        inline Vector3 &operator*=(const Vector3 &b)
        {
            x *= b.x, y *= b.y, z *= b.z;
            return (*this);
        }
        inline Vector3 &operator/=(const Vector3 &b)
        {
            x /= b.x, y /= b.y, z /= b.z;
            return (*this);
        }
        inline Vector3 &operator+=(const FLOAT_ &n)
        {
            x += n, y += n, z += n;
            return (*this);
        }
        inline Vector3 &operator-=(const FLOAT_ &n)
        {
            x -= n, y -= n, z -= n;
            return (*this);
        }
        inline Vector3 &operator*=(const FLOAT_ &n)
        {
            x *= n, y *= n, z *= n;
            return (*this);
        }
        inline Vector3 &operator/=(const FLOAT_ &n)
        {
            x /= n, y /= n, z /= n;
            return (*this);
        }
        inline Vector3 operator+(const Vector3 &b) const { return Vector3(*this) += b; }
        inline Vector3 operator-(const Vector3 &b) const { return Vector3(*this) -= b; }
        inline Vector3 operator*(const Vector3 &b) const { return Vector3(*this) *= b; }
        inline Vector3 operator/(const Vector3 &b) const { return Vector3(*this) /= b; }
        inline Vector3 operator+(const FLOAT_ &n) const { return Vector3(*this) += n; }
        inline Vector3 operator-(const FLOAT_ &n) const { return Vector3(*this) -= n; }
        inline Vector3 operator*(const FLOAT_ &n) const { return Vector3(*this) *= n; }
        inline Vector3 operator/(const FLOAT_ &n) const { return Vector3(*this) /= n; }
        inline friend Vector3 operator+(const FLOAT_ &n, const Vector3 &b) { return Vector3(n) += b; }
        inline friend Vector3 operator-(const FLOAT_ &n, const Vector3 &b) { return Vector3(n) -= b; }
        inline friend Vector3 operator*(const FLOAT_ &n, const Vector3 &b) { return Vector3(n) *= b; }
        inline friend Vector3 operator/(const FLOAT_ &n, const Vector3 &b) { return Vector3(n) /= b; }

        /* 向量的平方模 */
        inline FLOAT_ sqrMagnitude() const { return pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2); }
        /* 向量的模 */
        inline FLOAT_ magnitude() const { return sqrt(this->sqrMagnitude()); }
        /*判等*/
        inline bool equals(const Vector3 &b) const { return (*this) == b; }
        /* 向量单位化 */
        inline void Normalize()
        {
            FLOAT_ _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
            this->z /= _m;
        }

        /*转为字符串*/
        inline std::string ToString() const
        {
            std::ostringstream ostr;
            ostr << "Vector3(" << this->x << ", " << this->y << ", " << this->z << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量 */
        inline Vector3 normalized() const
        {
            FLOAT_ _m = this->magnitude();
            return Vector3(this->x / _m, this->y / _m, this->z / _m);
        }
        /* 距离 */
        inline static FLOAT_ Distance(const Vector3 &a, const Vector3 &b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        inline static Vector3 LerpUnclamped(const Vector3 &a, const Vector3 &b, const FLOAT_ &t) { return a + (b - a) * t; }

        /* 拿它的垂直向量（逆时针旋转90°） */
        inline static Vector3 Perpendicular(const Vector3 &inDirection) { return Vector3(-inDirection.y, inDirection.x, 0); }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        inline static Vector3 Reflect(const Vector3 &inDirection, const Vector3 &inNormal) { return inDirection - 2 * Vector3::Dot(inDirection, inNormal) * inNormal; }

        /* 点积 */
        inline static FLOAT_ Dot(const Vector3 &lhs, const Vector3 &rhs) const { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
        /* 叉积 */
        inline static Vector3 Cross(const Vector3 &lhs, const Vector3 &rhs) const { return Vector3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }

        /*无符号弧度夹角*/
        inline static FLOAT_ Rad(const Vector3 &from, const Vector3 &to) { return acos(Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /*无符号角度夹角*/
        inline static FLOAT_ Angle(const Vector3 &from, const Vector3 &to) { return Rad(from, to) * 180 / PI; }

        /*返回该方向上最大不超过maxLength长度的向量*/
        inline static Vector3 ClampMagnitude(const Vector3 &vector, const FLOAT_ &maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        inline static Vector3 Max(const Vector3 &lhs, const Vector3 &rhs) { return Vector3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)); }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        inline static Vector3 Min(const Vector3 &lhs, const Vector3 &rhs) { return Vector3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)); }

        /*获得vector在onNormal(请自己单位化)方向的投影*/
        inline static Vector3 Project(const Vector3 &vector, const Vector3 &onNormal) { return vector.magnitude() * cos(Rad(vector, onNormal)) * onNormal; }

        /*正交化：将两个向量单位化，并调整切线位置使之垂直于法向*/
        inline static void OrthoNormalize(Vector3 &normal, Vector3 &tangent)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
        }

        /*正交化：将三个向量单位化，并调整使之两两垂直*/
        inline static void OrthoNormalize(Vector3 &normal, Vector3 &tangent, Vector3 &binormal)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
            binormal -= Project(binormal, normal);
            binormal -= Project(binormal, tangent);
            binormal.Normalize();
        }

        /*获得vector在以planeNormal为法向量的平面的投影*/
        inline static Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal)
        {
            FLOAT_ mag = vector.magnitude();
            FLOAT_ s = Rad(vector, planeNormal);
            OrthoNormalize(planeNormal, vector);
            return mag * sin(s) * vector;
        }

        /*罗德里格旋转公式，获得current绕轴normal(请自己单位化)旋转degree度（默认角度）的向量，右手螺旋意义*/
        inline static Vector3 Rotate(const Vector3 &current, const Vector3 &normal, const FLOAT_ &degree, bool use_degree = 1)
        {
            FLOAT_ r = use_degree ? degree / 180 * PI : degree;
            FLOAT_ c = cos(r);
            return c * current + (1.0 - c) * Dot(normal, current) * normal + Cross(sin(r) * normal, current);
        }

        /*将current向target转向degree度，如果大于夹角则返回target方向长度为current的向量*/
        inline static Vector3 RotateTo(const Vector3 &current, const Vector3 &target, const FLOAT_ &degree, bool use_degree = 1)
        {
            FLOAT_ r = use_degree ? degree / 180 * PI : degree;
            if (r >= Rad(current, target))
                return current.magnitude() / target.magnitude() * target;
            else
            {
                // FLOAT_ mag = current.magnitude();
                Vector3 nm = Cross(current, target).normalized();
                return Rotate(current, nm, r);
            }
        }

        /*球面插值*/
        inline static Vector3 SlerpUnclamped(const Vector3 &a, const Vector3 &b, const FLOAT_ &t)
        {
            Vector3 rot = RotateTo(a, b, Rad(a, b) * t, false);
            FLOAT_ l = b.magnitude() * t + a.magnitude() * (1 - t);
            return rot.normalized() * l;
        }

        /*根据经纬，拿一个单位化的三维向量，以北纬和东经为正*/
        inline static Vector3 FromLongitudeAndLatitude(const FLOAT_ &longitude, const FLOAT_ &latitude)
        {
            Vector3 lat = Rotate(Vector3(1, 0, 0), Vector3(0, -1, 0), latitude);
            return Rotate(lat, Vector3(0, 0, 1), longitude);
        }

        /*球坐标转换为xyz型三维向量*/
        inline static Vector3 FromSphericalCoordinate(const Vector3 &spherical, bool use_degree = 1) { return FromSphericalCoordinate(spherical.x, spherical.y, spherical.z, use_degree); }
        /*球坐标转换为xyz型三维向量，半径r，theta倾斜角（纬度），phi方位角（经度），默认输出角度*/
        inline static Vector3 FromSphericalCoordinate(const FLOAT_ &r, FLOAT_ theta, FLOAT_ phi, bool use_degree = 1)
        {
            theta = use_degree ? theta / 180 * PI : theta;
            phi = use_degree ? phi / 180 * PI : phi;
            return Vector3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        inline static Vector3 ToSphericalCoordinate(const Vector3 &coordinate, bool use_degree = 1)
        {
            FLOAT_ r = coordinate.magnitude();
            return Vector3(
                r,
                acos(coordinate.z / r) * (use_degree ? 180.0 / PI : 1),
                atan2(coordinate.y, coordinate.x) * (use_degree ? 180.0 / PI : 1));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        inline Vector3 toSphericalCoordinate(bool use_degree = 1) { return ToSphericalCoordinate(*this, use_degree); }
    };

}