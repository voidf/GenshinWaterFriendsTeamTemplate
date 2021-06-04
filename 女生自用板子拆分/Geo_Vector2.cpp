#include "Geo_Base.cpp"

namespace Geometry
{

    struct Vector2
    {
        FLOAT_ x, y;
        Vector2(FLOAT_ _x, FLOAT_ _y) : x(_x), y(_y) {}
        Vector2(FLOAT_ n) : x(n), y(n) {}
        Vector2() : x(0.0), y(0.0) {}
        Vector2 &operator=(Vector2 b)
        {
            this->x = b.x;
            this->y = b.y;
            return *this;
        }
        bool operator==(Vector2 b) { return round_compare(this->x, b.x) and round_compare(this->y, b.y); }
        bool operator!=(Vector2 b) { return not((*this) == b); }
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
            case 'x':
                return this->x;
                break;
            case 'y':
                return this->y;
                break;
            default:
                throw "无法理解除0,1外的索引";
                break;
            }
        }
        friend std::ostream &operator<<(std::ostream &o, Vector2 v)
        {
            o << v.ToString();
            return o;
        }
        friend Vector2 operator*(FLOAT_ n, Vector2 v) { return Vector2(v.x * n, v.y * n); }
        friend Vector2 operator/(FLOAT_ n, Vector2 v) { return Vector2(n / v.x, n / v.y); }
        Vector2 operator-() { return Vector2(-(this->x), -(this->y)); }
        Vector2 operator+(Vector2 b) { return Vector2(this->x + b.x, this->y + b.y); }
        Vector2 operator-(Vector2 b) { return (*this) + (-b); }
        Vector2 operator*(FLOAT_ n) { return Vector2(this->x * n, this->y * n); }
        Vector2 operator*(Vector2 b) { return Vector2(this->x * b.x, this->y * b.y); }
        Vector2 operator/(FLOAT_ n) { return (*this) * (FLOAT_(1) / n); }
        Vector2 operator/(Vector2 b) { return (*this) * (FLOAT_(1) / b); }
        Vector2 operator+=(Vector2 b) { return (*this) = (*this) + b; }

        bool operator<(Vector2 b) const { return this->x < b.x or this->x == b.x and this->y < b.y; }

        /* 向量的平方模 */
        FLOAT_ sqrMagnitude() { return pow(this->x, 2) + pow(this->y, 2); }
        /* 向量的模 */
        FLOAT_ magnitude() { return sqrt(this->sqrMagnitude()); }
        /*判等*/
        bool equals(Vector2 b) { return (*this) == b; }

        /*用极坐标换算笛卡尔坐标*/
        static Vector2 fromPolarCoordinate(Vector2 v, bool use_degree = 1)
        {
            return v.toCartesianCoordinate(use_degree);
        }

        /*转为笛卡尔坐标*/
        Vector2 toCartesianCoordinate(bool use_degree = 1)
        {
            return Vector2(
                x * cos(y * (use_degree ? PI / 180.0 : 1)),
                x * sin(y * (use_degree ? PI / 180.0 : 1)));
        }
        /*转为极坐标*/
        Vector2 toPolarCoordinate(bool use_degree = 1)
        {
            return Vector2(
                magnitude(),
                toPolarAngle(use_degree));
        }

        /*获取极角*/
        FLOAT_ toPolarAngle(bool use_degree = 1)
        {
            return atan2(y, x) * (use_degree ? 180.0 / PI : 1);
        }

        /*转为极坐标*/
        static Vector2 ToPolarCoordinate(Vector2 coordinate, bool use_degree = 1) { return coordinate.toPolarCoordinate(use_degree); }

        static bool Equals(Vector2 a, Vector2 b) { return a == b; }
        /* 向量单位化 */
        void Normalize()
        {
            FLOAT_ _m = this->magnitude();
            this->x /= _m;
            this->y /= _m;
        }
        /*设置值*/
        void Set(FLOAT_ newX, FLOAT_ newY)
        {
            this->x = newX;
            this->y = newY;
        }
        /*转为字符串*/
        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Vector2(" << this->x << ", " << this->y << ")";
            return ostr.str();
        }

        /* 返回与该向量方向同向的单位向量 */
        Vector2 normalized()
        {
            FLOAT_ _m = this->magnitude();
            return Vector2(this->x / _m, this->y / _m);
        }
        // FLOAT_ Distance(Vector2 b) { return ((*this) - b).magnitude(); }
        /* 距离 */
        static FLOAT_ Distance(Vector2 a, Vector2 b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static Vector2 LerpUnclamped(Vector2 a, Vector2 b, FLOAT_ t)
        {
            Vector2 c = b - a;
            return a + c * t;
        }

        /* 向量线性插值,t限制于[0,1] */
        static Vector2 Lerp(Vector2 a, Vector2 b, FLOAT_ t)
        {
            t = max(FLOAT_(0), t);
            t = min(FLOAT_(1), t);
            return Vector2::LerpUnclamped(a, b, t);
        }

        /* 向量线性插值,t限制于(-无穷,1] */
        static Vector2 MoveTowards(Vector2 current, Vector2 target, FLOAT_ maxDistanceDelta)
        {
            maxDistanceDelta = min(FLOAT_(1), maxDistanceDelta);
            return Vector2::LerpUnclamped(current, target, maxDistanceDelta);
        }

        /* 拿它的垂直向量（逆时针旋转90°） */
        static Vector2 Perpendicular(Vector2 inDirection)
        {
            return Vector2(-inDirection.y, inDirection.x);
        }
        /*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
        static Vector2 Reflect(Vector2 inDirection, Vector2 inNormal)
        {
            return inDirection - 2 * Vector2::Dot(inDirection, inNormal) * inNormal;
        }

        /* 点积 */
        static FLOAT_ Dot(Vector2 lhs, Vector2 rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y;
        }
        /* 叉积 */
        static FLOAT_ Cross(Vector2 lhs, Vector2 rhs) { return lhs.x * rhs.y - lhs.y * rhs.x; }

        /* 对位相乘罢了 */
        static Vector2 Scale(Vector2 a, Vector2 b) { return Vector2(a.x * b.x, a.y * b.y); }

        /* 对位相乘罢了 */
        Vector2 Scale(Vector2 scale) { return (*this) * scale; }

        /*有符号弧度夹角*/
        static FLOAT_ SignedRad(Vector2 from, Vector2 to) { return atan2(Vector2::Cross(from, to), Vector2::Dot(from, to)); }
        /*无符号弧度夹角*/
        static FLOAT_ Rad(Vector2 from, Vector2 to) { return abs(Vector2::SignedRad(from, to)); }
        /*有符号角度夹角*/
        static FLOAT_ SignedAngle(Vector2 from, Vector2 to)
        {
            // return acos(Vector2::Dot(from, to) / (from.magnitude() * to.magnitude()));
            return Vector2::SignedRad(from, to) * 180.0 / PI;
        }
        /*无符号角度夹角*/
        static FLOAT_ Angle(Vector2 from, Vector2 to) { return abs(Vector2::SignedAngle(from, to)); }

        /*返回该方向上最大不超过maxLength长度的向量*/
        static Vector2 ClampMagnitude(Vector2 vector, FLOAT_ maxLength)
        {
            if (vector.magnitude() <= maxLength)
                return vector;
            else
                return vector.normalized() * maxLength;
        }
        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static Vector2 Max(Vector2 lhs, Vector2 rhs)
        {
            return Vector2(max(lhs.x, rhs.x), max(lhs.y, rhs.y));
        }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static Vector2 Min(Vector2 lhs, Vector2 rhs)
        {
            return Vector2(min(lhs.x, rhs.x), min(lhs.y, rhs.y));
        }

        /*获得vector在onNormal方向的投影*/
        static Vector2 Project(Vector2 vector, Vector2 onNormal)
        {
            return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal;
        }
    };
    struct PolarSortCmp
    {
        bool operator()(Vector2 &a, Vector2 &b) { return a.toPolarAngle(0) < b.toPolarAngle(0); }
    };

    struct CrossSortCmp
    {
        bool operator()(Vector2 &a, Vector2 &b)
        {
            return Vector2::Cross(a, b) > 0;
        }
    };

}