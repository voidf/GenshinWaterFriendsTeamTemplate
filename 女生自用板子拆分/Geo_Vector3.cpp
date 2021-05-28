#include "Geo_Base.cpp"

namespace Geometry
{
    
    struct Vector3 // 三维向量
    {
        FLOAT_ x, y, z;
        Vector3(FLOAT_ _x, FLOAT_ _y, FLOAT_ _z) : x(_x), y(_y), z(_z) {}
        Vector3(FLOAT_ n) : x(n), y(n), z(n) {}
        Vector3(Vector2 &b) : x(b.x), y(b.y), z(0.0) {}
        // Vector3(Vector2 &&b) : x(b.x), y(b.y), z(0.0) {}
        Vector3() : x(0.0), y(0.0), z(0.0) {}
        Vector3 &operator=(Vector3 b)
        {
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            return *this;
        }
        bool operator==(Vector3 b) { return round_compare(this->x, b.x) and round_compare(this->y, b.y) and round_compare(this->z, b.z); }
        bool operator!=(Vector3 b) { return not((*this) == b); }
        FLOAT_ &operator[](int ind)
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
        friend std::ostream &operator<<(std::ostream &o, Vector3 v)
        {
            o << v.ToString();
            return o;
        }
        friend Vector3 operator*(FLOAT_ n, Vector3 v) { return Vector3(v.x * n, v.y * n, v.z * n); }
        friend Vector3 operator/(FLOAT_ n, Vector3 v) { return Vector3(n / v.x, n / v.y, n / v.z); }
        Vector3 operator-() { return Vector3(-(this->x), -(this->y), -(this->z)); }
        Vector3 operator+(Vector3 b) { return Vector3(this->x + b.x, this->y + b.y, this->z + b.z); }
        Vector3 operator-(Vector3 b) { return (*this) + (-b); }
        Vector3 operator*(FLOAT_ n) { return Vector3(this->x * n, this->y * n, this->z * n); }
        Vector3 operator*(Vector3 b) { return Vector3(this->x * b.x, this->y * b.y, this->z * b.z); }
        Vector3 operator/(FLOAT_ n) { return (*this) * (FLOAT_(1) / n); }
        Vector3 operator/(Vector3 b) { return (*this) * (FLOAT_(1) / b); }
        Vector3 operator-=(Vector3 b) { return (*this) = (*this) - b; }

        /* 向量的平方模 */
        FLOAT_ sqrMagnitude() { return pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2); }
        /* 向量的模 */
        FLOAT_ magnitude() { return sqrt(this->sqrMagnitude()); }
        /*判等*/
        bool equals(Vector3 b) { return (*this) == b; }
        static bool Equals(Vector3 a, Vector3 b) { return a == b; }
        /* 向量单位化 */
        void Normalize()
        {
            FLOAT_ _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
            this->z /= _m;
        }
        /*设置值*/
        void Set(FLOAT_ newX, FLOAT_ newY, FLOAT_ newZ)
        {
            this->x = newX;
            this->y = newY;
            this->z = newZ;
        }
        /*转为字符串*/
        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Vector3(" << this->x << ", " << this->y << ", " << this->z << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量 */
        Vector3 normalized()
        {
            FLOAT_ _m = this->magnitude();
            return Vector3(this->x / _m, this->y / _m, this->z / _m);
        }
        // FLOAT_ Distance(Vector2 b) { return ((*this) - b).magnitude(); }
        /* 距离 */
        static FLOAT_ Distance(Vector3 a, Vector3 b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static Vector3 LerpUnclamped(Vector3 a, Vector3 b, FLOAT_ t) { return a + (b - a) * t; }

        /* 向量线性插值,t限制于[0,1] */
        static Vector3 Lerp(Vector3 a, Vector3 b, FLOAT_ t)
        {
            t = max(FLOAT_(0), t);
            t = min(FLOAT_(1), t);
            return Vector3::LerpUnclamped(a, b, t);
        }

        /* 向量线性插值,t限制于(-无穷,1] */
        static Vector3 MoveTowards(Vector3 current, Vector3 target, FLOAT_ maxDistanceDelta)
        {
            maxDistanceDelta = min(FLOAT_(1), maxDistanceDelta);
            return Vector3::LerpUnclamped(current, target, maxDistanceDelta);
        }

        /* 拿它的垂直向量（逆时针旋转90°） */
        static Vector3 Perpendicular(Vector3 inDirection)
        {
            return Vector3(-inDirection.y, inDirection.x, 0);
        }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        static Vector3 Reflect(Vector3 inDirection, Vector3 inNormal)
        {
            return inDirection - 2 * Vector3::Dot(inDirection, inNormal) * inNormal;
        }

        /* 点积 */
        static FLOAT_ Dot(Vector3 lhs, Vector3 rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }
        /* 叉积 */
        static Vector3 Cross(Vector3 lhs, Vector3 rhs) { return Vector3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }

        /* 对位相乘罢了 */
        static Vector3 Scale(Vector3 a, Vector3 b) { return a * b; }

        /* 对位相乘罢了 */
        Vector3 Scale(Vector3 scale) { return (*this) * scale; }

        /*无符号弧度夹角*/
        static FLOAT_ Rad(Vector3 from, Vector3 to) { return acos(Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /*无符号角度夹角*/
        static FLOAT_ Angle(Vector3 from, Vector3 to) { return Rad(from, to) * 180 / PI; }

        /*返回该方向上最大不超过maxLength长度的向量*/
        static Vector3 ClampMagnitude(Vector3 vector, FLOAT_ maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static Vector3 Max(Vector3 lhs, Vector3 rhs)
        {
            return Vector3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z));
        }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static Vector3 Min(Vector3 lhs, Vector3 rhs)
        {
            return Vector3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z));
        }

        /*获得vector在onNormal(请自己单位化)方向的投影*/
        static Vector3 Project(Vector3 vector, Vector3 onNormal)
        {
            return vector.magnitude() * cos(Rad(vector, onNormal)) * onNormal;
        }

        /*将两个向量单位化，并调整切线位置使之垂直于法向*/
        static void OrthoNormalize(Vector3 &normal, Vector3 &tangent)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
        }

        /*将三个向量单位化，并调整使之两两垂直*/
        static void OrthoNormalize(Vector3 &normal, Vector3 &tangent, Vector3 &binormal)
        {
            normal.Normalize();
            tangent = tangent - Project(tangent, normal);
            tangent.Normalize();
            binormal -= Project(binormal, normal);
            binormal -= Project(binormal, tangent);
            binormal.Normalize();
        }

        /*获得vector在以planeNormal为法向量的平面的投影*/
        static Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal)
        {
            FLOAT_ mag = vector.magnitude();
            FLOAT_ s = Rad(vector, planeNormal);
            OrthoNormalize(planeNormal, vector);
            return mag * sin(s) * vector;
        }

        /*罗德里格旋转公式，获得current绕轴normal(请自己单位化)旋转degree度（默认角度）的向量，右手螺旋意义*/
        static Vector3 Rotate(Vector3 current, Vector3 normal, FLOAT_ degree, bool use_degree = 1)
        {
            FLOAT_ r = use_degree ? degree / 180 * PI : degree;
            FLOAT_ c = cos(r);
            return c * current + (1.0 - c) * Dot(normal, current) * normal + Cross(sin(r) * normal, current);
        }

        /*将current向target转向degree度，如果大于夹角则返回target方向长度为current的向量*/
        static Vector3 RotateTo(Vector3 current, Vector3 target, FLOAT_ degree, bool use_degree = 1)
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
        static Vector3 SlerpUnclamped(Vector3 a, Vector3 b, FLOAT_ t)
        {
            Vector3 rot = RotateTo(a, b, Rad(a, b) * t, false);
            FLOAT_ l = b.magnitude() * t + a.magnitude() * (1 - t);
            return rot.normalized() * l;
        }

        /*球面插值，t限制于[0,1]版本*/
        static Vector3 Slerp(Vector3 a, Vector3 b, FLOAT_ t)
        {
            if (t <= 0)
                return a;
            if (t >= 1)
                return b;
            return SlerpUnclamped(a, b, t);
        }
        /*根据经纬，拿一个单位化的三维向量，以北纬和东经为正*/
        static Vector3 FromLongitudeAndLatitude(FLOAT_ longitude, FLOAT_ latitude)
        {
            Vector3 lat = Rotate(Vector3(1, 0, 0), Vector3(0, -1, 0), latitude);
            return Rotate(lat, Vector3(0, 0, 1), longitude);
        }

        /*球坐标转换为xyz型三维向量*/
        static Vector3 FromSphericalCoordinate(Vector3 spherical, bool use_degree = 1) { return FromSphericalCoordinate(spherical.x, spherical.y, spherical.z, use_degree); }
        /*球坐标转换为xyz型三维向量，半径r，theta倾斜角（纬度），phi方位角（经度），默认输出角度*/
        static Vector3 FromSphericalCoordinate(FLOAT_ r, FLOAT_ theta, FLOAT_ phi, bool use_degree = 1)
        {
            theta = use_degree ? theta / 180 * PI : theta;
            phi = use_degree ? phi / 180 * PI : phi;
            return Vector3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        static Vector3 ToSphericalCoordinate(Vector3 coordinate, bool use_degree = 1)
        {
            FLOAT_ r = coordinate.magnitude();
            return Vector3(
                r,
                acos(coordinate.z / r) * (use_degree ? 180.0 / PI : 1),
                atan2(coordinate.y, coordinate.x) * (use_degree ? 180.0 / PI : 1));
        }
        /*直角坐标转换为球坐标，默认输出角度*/
        Vector3 toSphericalCoordinate(bool use_degree = 1) { return ToSphericalCoordinate(*this, use_degree); }

    };

}