# 必要的头

```cpp
namespace Geometry
{
    using FLOAT_ = double;

    constexpr const FLOAT_ Infinity = INFINITY;
    const FLOAT_ decimal_round = 1e-8; // 精度参数

    const FLOAT_ DEC = 1.0 / decimal_round;
    const double smooth_const2 = 0.479999989271164; // 二次项平滑系数
    const double smooth_const3 = 0.234999999403954; // 三次项平滑系数

    const FLOAT_ PI = acos(-1);
    bool round_compare(FLOAT_ a, FLOAT_ b) { return round(DEC * a) == round(DEC * b); }
    FLOAT_ Round(FLOAT_ a) { return round(DEC * a) / DEC; }
}
```

# 分数类

```cpp
template <typename PrecisionType = long long>
struct Fraction
{
    PrecisionType upper, lower;

    Fraction(PrecisionType u = 0, PrecisionType l = 1)
    {
        upper = u;
        lower = l;
    }
    void normalize()
    {
        if (upper)
        {
            PrecisionType g = abs(std::__gcd(upper, lower));
            upper /= g;
            lower /= g;
        }
        else
            lower = 1;
        if (lower < 0)
        {
            lower = -lower;
            upper = -upper;
        }
    }
    long double ToFloat() { return (long double)upper / (long double)lower; }
    bool operator==(Fraction b) { return upper * b.lower == lower * b.upper; }
    bool operator>(Fraction b) { return upper * b.lower > lower * b.upper; }
    bool operator<(Fraction b) { return upper * b.lower < lower * b.upper; }
    bool operator<=(Fraction b) { return !(*this > b); }
    bool operator>=(Fraction b) { return !(*this < b); }
    bool operator!=(Fraction b) { return !(*this == b); }
    Fraction operator-() { return Fraction(-upper, lower); }
    Fraction operator+(Fraction b) { return Fraction(upper * b.lower + b.upper * lower, lower * b.lower); }
    Fraction operator-(Fraction b) { return (*this) + (-b); }
    Fraction operator*(Fraction b) { return Fraction(upper * b.upper, lower * b.lower); }
    Fraction operator/(Fraction b) { return Fraction(upper * b.lower, lower * b.upper); }
    Fraction &operator+=(Fraction b)
    {
        *this = *this + b;
        this->normalize();
        return *this;
    }
    Fraction &operator-=(Fraction b)
    {
        *this = *this - b;
        this->normalize();
        return *this;
    }
    Fraction &operator*=(Fraction b)
    {
        *this = *this * b;
        this->normalize();
        return *this;
    }
    Fraction &operator/=(Fraction b)
    {
        *this = *this / b;
        this->normalize();
        return *this;
    }
    friend Fraction fabs(Fraction a) { return Fraction(abs(a.upper), abs(a.lower)); }
    std::string to_string() { return lower == 1 ? std::to_string(upper) : std::to_string(upper) + '/' + std::to_string(lower); }
    friend std::ostream &operator<<(std::ostream &o, Fraction a)
    {
        return o << "Fraction(" << std::to_string(a.upper) << ", " << std::to_string(a.lower) << ")";
    }
    friend std::istream &operator>>(std::istream &i, Fraction &a)
    {
        char slash;
        return i >> a.upper >> slash >> a.lower;
    }
    friend isfinite(Fraction a) { return a.lower != 0; }
    void set_value(PrecisionType u, PrecisionType d = 1) { upper = u, lower = d; }
};
```

# 二维向量

```cpp
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

    bool operator<(Vector2 b) { return this->x < b.x or this->x == b.x and this->y < b.y; }

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
    /*极坐标排序比较器*/
    static std::function<bool(Vector2 &, Vector2 &)> PolarSortCmp = [](Vector2 &a, Vector2 &b) -> bool
    {
        return a.toPolarAngle(0) < b.toPolarAngle(0);
    };

    /*叉乘排序比较器*/
    static std::function<bool(Vector2 &, Vector2 &)> CrossSortCmp = [](Vector2 &a, Vector2 &b) -> bool
    {
        return Cross(a, b) > 0;
    };

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
```

# 三维向量

```cpp
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
```

# N维向量

```cpp
template <typename VALUETYPE = FLOAT_>
struct VectorN : std::vector<VALUETYPE>
{
    void init(int siz, VALUETYPE default_val = 0)
    {
        this->clear();
        this->assign(siz, default_val);
        this->resize(siz);
    }
    VectorN(int siz, VALUETYPE default_val = 0) { init(siz, default_val); }
    VectorN() = default;

    bool any()
    {
        for (auto &i : *this)
        {
            if (i != 0)
                return true;
            else if (!isfinite(i))
                return false;
        }
        return false;
    }
    bool is_all_nan()
    {
        for (auto &i : *this)
        {
            if (!isnan(i))
                return false;
        }
        return true;
    }

    void rconcat(VectorN &&r)
    {
        this->insert(this->end(), r.begin(), r.end());
    }
    void rconcat(VectorN &r) { rconcat(std::move(r)); }

    void lerase(int ctr)
    {
        assert(this->size() >= ctr);
        this->erase(this->begin(), this->begin() + ctr);
    }

    

    // 四则运算 和单个数

    VectorN operator*(VALUETYPE &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] * operand;
        return ret;
    }
    friend VectorN operator*(VALUETYPE &&operand, VectorN &r)
    {
        VectorN ret(r.size());
        for (int i = 0; i < r.size(); i++)
            ret[i] = operand * r[i];
        return ret;
    }

    VectorN operator/(VALUETYPE &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] / operand;
        return ret;
    }
    friend VectorN operator/(VALUETYPE &&operand, VectorN &r)
    {
        VectorN ret(r.size());
        for (int i = 0; i < r.size(); i++)
            ret[i] = operand / r[i];
        return ret;
    }

    VectorN operator+(VALUETYPE &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] + operand;
        return ret;
    }
    friend VectorN operator+(VALUETYPE &&operand, VectorN &r)
    {
        VectorN ret(r.size());
        for (int i = 0; i < r.size(); i++)
            ret[i] = operand + r[i];
        return ret;
    }

    VectorN operator-(VALUETYPE &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] - operand;
        return ret;
    }
    friend VectorN operator-(VALUETYPE &&operand, VectorN &r)
    {
        VectorN ret(r.size());
        for (int i = 0; i < r.size(); i++)
            ret[i] = operand - r[i];
        return ret;
    }

    
    // 四则运算 和同类

    VectorN operator+(VectorN &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] + operand[i];
        return ret;
    }

    VectorN operator-(VectorN &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] - operand[i];
        return ret;
    }

    VectorN operator*(VectorN &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] * operand[i];
        return ret;
    }

    VectorN operator/(VectorN &&operand)
    {
        VectorN ret(this->size());
        for (int i = 0; i < this->size(); i++)
            ret[i] = (*this)[i] / operand[i];
        return ret;
    }

    // 结束

    std::string ToString()
    {
        std::ostringstream ostr;
        for (int i = 0; i < this->size(); i++)
        {
            ostr << std::setw(8) << std::scientific << std::right;
            ostr << (*this)[i] << (i == this->size() - 1 ? "\n" : ", ");
        }
        return ostr.str();
    }

    /* 向量的平方模 */
    VALUETYPE sqrMagnitude()
    {
        VALUETYPE res = 0;
        for (auto &i : (*this))
            res += i * i;
        return res;
    }

    /* 向量的模 */
    VALUETYPE magnitude() { return sqrt(this->sqrMagnitude()); }

    /* 向量单位化 */
    void Normalize()
    {
        VALUETYPE _m = this->magnitude();
        for (auto &i : (*this))
            i /= _m;
    }

    /* 返回与该向量方向同向的单位向量 */
    VectorN normalized()
    {
        VectorN ret(*this);
        ret.Normalize();
        return ret;
    }

    /* 距离 */
    static VALUETYPE Distance(VectorN &a, VectorN &b) { return (a - b).magnitude(); }

    /*向量线性插值*/
    static VectorN LerpUnclamped(VectorN &a, VectorN &b, VALUETYPE t) { return a + (b - a) * t; }

    /* 点积 */

    static VALUETYPE Dot(VectorN &&lhs, VectorN &&rhs)
    {
        VALUETYPE ans = 0;
        for (auto i = 0; i < lhs._len; i++)
            ans += lhs.data[i] * rhs.data[i];
        return ans;
    }

    /*无符号弧度夹角*/
    static VALUETYPE Rad(VectorN &from, VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }

    /*无符号角度夹角*/
    static VALUETYPE Angle(VectorN &from, VectorN &to) { return Rad(from, to) * 180.0 / PI; }

    /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
    static VectorN Max(VectorN lhs, VectorN &&rhs)
    {
        for (auto &&i : range(lhs._len))
            lhs.data[i] = std::max(lhs.data[i], rhs.data[i]);
        return lhs;
    }

    /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
    static VectorN Min(VectorN lhs, VectorN &&rhs)
    {
        for (auto &&i : range(lhs._len))
            lhs.data[i] = std::min(lhs.data[i], rhs.data[i]);
        return lhs;
    }

    /*获得vector在onNormal方向的投影*/
    static VectorN Project(VectorN &vector, VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }

};
```

# 矩阵

```cpp
template <typename VALUETYPE = FLOAT_>
struct Matrix : VectorN<VectorN<VALUETYPE>>
{
    int ROW, COL;

    std::string ToString()
    {
        std::ostringstream ostr;
        ostr << "Matrix" << ROW << "x" << COL << "[\n";
        for (auto &i : *this)
            ostr << '\t' << i.ToString();
        ostr << "]";
        return ostr.str();
    }

    Matrix(VectorN<VectorN<VALUETYPE>> &&v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}
    Matrix(VectorN<VectorN<VALUETYPE>> &v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}

    Matrix(int r, int c, VALUETYPE default_val = 0) : ROW(r), COL(c)
    {
        this->resize(r);
        for (r--; r >= 0; r--)
            (*this)[r].init(c, default_val);
    }
    Matrix() = default;

    /*交换两行*/
    void swap_rows(int from, int to)
    {
        std::swap((*this)[from], (*this)[to]);
    }

    /*化为上三角矩阵*/
    void triangularify(bool unitriangularify = false)
    {
        int mx;
        int done_rows = 0;
        for (int j = 0; j < COL; j++) // 化为上三角
        {
            mx = done_rows;
            for (int i = done_rows + 1; i < ROW; i++)
            {
                if (fabs((*this)[i][j]) > fabs((*this)[mx][j]))
                    mx = i;
            }
            if ((*this)[mx][j] == 0)
                continue;
            if (mx != done_rows)
                swap_rows(mx, done_rows);

            for (int i = done_rows + 1; i < ROW; i++)
            {
                VALUETYPE tmp = (*this)[i][j] / (*this)[done_rows][j];
                if (tmp != 0)
                    (*this)[i] -= (*this)[done_rows] * tmp;
            }
            if (unitriangularify)
            {
                auto tmp = (*this)[done_rows][j];
                (*this)[done_rows] /= tmp; // 因为用了引用，这里得拷贝暂存
            }
            done_rows++;
            if (done_rows == ROW)
                break;
        }
    }

    /*化为上三角矩阵，模意义版*/
    void triangularify(long long mod, bool unitriangularify = false)
    {
        int mx;
        int done_rows = 0;
        for (int j = 0; j < COL; j++) // 化为上三角
        {
            mx = done_rows;

            if ((*this)[done_rows][j] < 0)
                (*this)[done_rows][j] = ((*this)[done_rows][j] % mod + mod) % mod;

            for (int i = done_rows + 1; i < ROW; i++)
            {
                if ((*this)[i][j] < 0)
                    (*this)[i][j] = ((*this)[i][j] % mod + mod) % mod;
                if ((*this)[i][j] > (*this)[mx][j])
                    mx = i;
            }
            if ((*this)[mx][j] == 0)
                continue;

            if (mx != done_rows)
                swap_rows(mx, done_rows);

            for (int i = done_rows + 1; i < ROW; i++)
            {
                VALUETYPE tmp = (*this)[i][j] * inv((*this)[done_rows][j], mod) % mod;
                if (tmp != 0)
                {
                    (*this)[i] -= (*this)[done_rows] * tmp;
                    (*this)[i] %= mod;
                }
            }
            if (unitriangularify)
            {
                auto tmp = (*this)[done_rows][j];
                (*this)[done_rows] *= inv(tmp, mod);
                (*this)[done_rows] %= mod;
            }
            done_rows++;
            if (done_rows == ROW)
                break;
        }
    }

    void row_echelonify(long long mod = 0)
    {
        if (mod)
            triangularify(mod, true);
        else
            triangularify(true);
        int valid_pos = 1;
        for (int i = 1; i < ROW; i++)
        {
            while (valid_pos < COL and (*this)[i][valid_pos] == 0)
                valid_pos++;
            if (valid_pos == COL)
                break;
            for (int ii = i - 1; ii >= 0; ii--)
            {
                (*this)[ii] -= (*this)[i] * (*this)[ii][valid_pos];
                if (mod)
                    (*this)[ii] %= mod;
            }
        }
    }

    /*返回一个自身化为上三角矩阵的拷贝*/
    Matrix triangular(bool unitriangularify = false)
    {
        Matrix ret(*this);
        ret.triangularify(unitriangularify);
        return ret;
    }

    /*求秩，得先上三角化*/
    int _rank()
    {
        int res = 0;
        for (auto &i : (*this))
            res += i.any();
        return res;
    }

    /*求秩*/
    int rank() { return triangular()._rank(); }

    /*高斯消元解方程组*/
    bool solve()
    {
        if (COL != ROW + 1)
            throw "dimension error!";
        triangularify();
        // cerr << *this << endl;
        if (!(*this).back().any())
            return false;
        for (int i = ROW - 1; i >= 0; i--)
        {
            for (int j = i + 1; j < ROW; j++)
                (*this)[i][COL - 1] -= (*this)[i][j] * (*this)[j][COL - 1];
            if ((*this)[i][i] == 0)
                return false;
            (*this)[i][COL - 1] /= (*this)[i][i];
        }
        return true;
    }

    /*矩阵连接*/
    void rconcat(Matrix &&rhs)
    {
        COL += rhs.COL;
        for (int i = 0; i < ROW; i++)
        {
            (*this)[i].rconcat(rhs[i]);
        }
    }

    /*左截断*/
    void lerase(int ctr)
    {
        assert(COL >= ctr);
        COL -= ctr;
        for (int i = 0; i < ROW; i++)
        {
            (*this)[i].lerase(ctr);
        }
    }

    /*矩阵乘法*/
    Matrix dot(Matrix &&rhs, long long mod = 0)
    {
        if (this->COL != rhs.ROW)
            throw "Error at matrix multiply: lhs's column is not equal to rhs's row";
        Matrix ret(this->ROW, rhs.COL, true);
        for (int i = 0; i < ret.ROW; ++i)
            for (int k = 0; k < this->COL; ++k)
            {
                VALUETYPE &s = (*this)[i][k];
                for (int j = 0; j < ret.COL; ++j)
                {
                    ret[i][j] += s * rhs[k][j];
                    if (mod)
                        ret[i][j] %= mod;
                }
            }
        return ret;
    }
};
```
# 方阵

```cpp
template <typename VALUETYPE = FLOAT_>
struct SquareMatrix : Matrix<VALUETYPE>
{
    SquareMatrix(int siz, bool is_reset = false) : Matrix<VALUETYPE>(siz, siz, is_reset) {}
    SquareMatrix(Matrix<VALUETYPE> &&x) : Matrix<VALUETYPE>(x)
    {
        assert(x.COL == x.ROW);
    }
    static SquareMatrix eye(int siz)
    {
        SquareMatrix ret(siz, true);
        for (siz--; siz >= 0; siz--)
            ret[siz][siz] = 1;
        return ret;
    }

    SquareMatrix quick_power(long long p, long long mod = 0)
    {
        SquareMatrix ans = eye(this->ROW);
        SquareMatrix rhs(*this);
        while (p)
        {
            if (p & 1)
            {
                ans = ans.dot(rhs, mod);
            }
            rhs = rhs.dot(rhs, mod);
            p >>= 1;
        }
        return ans;
    }

    SquareMatrix inv(long long mod = 0)
    {
        Matrix<VALUETYPE> ret(*this);
        ret.rconcat(eye(this->ROW));
        ret.row_echelonify(mod);
        // cerr << ret << endl;
        for (int i = 0; i < this->ROW; i++)
        {
            if (ret[i][i] != 1)
                throw "Error at matrix inverse: cannot identify extended matrix";
        }
        ret.lerase(this->ROW);
        return ret;
    }
};
```

# 二维直线

```cpp
struct Line2
{
    FLOAT_ A, B, C;
    Line2(Vector2 u, Vector2 v) : A(u.y - v.y), B(v.x - u.x), C(u.y * (u.x - v.x) - u.x * (u.y - v.y))
    {
        if (u == v)
        {
            if (u.x)
            {
                A = 1;
                B = 0;
                C = -u.x;
            }
            else if (u.y)
            {
                A = 0;
                B = 1;
                C = -u.y;
            }
            else
            {
                A = 1;
                B = -1;
                C = 0;
            }
        }
    }
    Line2(FLOAT_ a, FLOAT_ b, FLOAT_ c) : A(a), B(b), C(c) {}
    std::string ToString()
    {
        std::ostringstream ostr;
        ostr << "Line2(" << this->A << ", " << this->B << ", " << this->C << ")";
        return ostr.str();
    }
    friend std::ostream &operator<<(std::ostream &o, Line2 v)
    {
        o << v.ToString();
        return o;
    }
    FLOAT_ k() { return -A / B; }
    FLOAT_ b() { return -C / B; }
    FLOAT_ x(FLOAT_ y) { return -(B * y + C) / A; }
    FLOAT_ y(FLOAT_ x) { return -(A * x + C) / B; }
    /*点到直线的距离*/
    FLOAT_ distToPoint(Vector2 p) { return abs(A * p.x + B * p.y + C / sqrt(A * A + B * B)); }
    /*直线距离公式，使用前先判平行*/
    static FLOAT_ Distance(Line2 a, Line2 b) { return abs(a.C - b.C) / sqrt(a.A * a.A + a.B * a.B); }
    /*判断平行*/
    static bool IsParallel(Line2 u, Line2 v)
    {
        bool f1 = round_compare(u.B, 0.0);
        bool f2 = round_compare(v.B, 0.0);
        if (f1 != f2)
            return false;
        return f1 or round_compare(u.k(), v.k());
    }

    /*单位化（？）*/
    void normalize()
    {
        FLOAT_ su = sqrt(A * A + B * B + C * C);
        if (A < 0)
            su = -su;
        else if (A == 0 and B < 0)
            su = -su;
        A /= su;
        B /= su;
        C /= su;
    }
    /*返回单位化后的直线*/
    Line2 normalized()
    {
        Line2 t(*this);
        t.normalize();
        return t;
    }

    bool operator==(Line2 v) { return round_compare(A, v.A) and round_compare(B, v.B) and round_compare(C, v.C); }
    bool operator!=(Line2 v) { return !(*this == v); }

    static bool IsSame(Line2 u, Line2 v)
    {
        return Line2::IsParallel(u, v) and round_compare(Distance(u.normalized(), v.normalized()), 0.0);
    }

    /*计算交点*/
    static Vector2 Intersect(Line2 u, Line2 v)
    {
        FLOAT_ tx = (u.B * v.C - v.B * u.C) / (v.B * u.A - u.B * v.A);
        FLOAT_ ty = (u.B != 0.0 ? -u.A * tx / u.B - u.C / u.B : -v.A * tx / v.B - v.C / v.B);
        return Vector2(tx, ty);
    }

    /*判断三个点是否共线*/
    static bool Collinear(Vector2 a, Vector2 b, Vector2 c)
    {
        Line2 l(a, b);
        return round_compare((l.A * c.x + l.B * c.y + l.C), 0.0);
    }
};
```

# 二维有向线段

```cpp
struct Segment2 : Line2 // 二维有向线段
{
    Vector2 from, to;
    Segment2(Vector2 a, Vector2 b) : Line2(a, b), from(a), to(b) {}
    Segment2(FLOAT_ x, FLOAT_ y, FLOAT_ X, FLOAT_ Y) : Line2(Vector2(x, y), Vector2(X, Y)), from(Vector2(x, y)), to(Vector2(X, Y)) {}
    bool is_online(Vector2 poi)
    {
        return round_compare((Vector2::Distance(poi, to) + Vector2::Distance(poi, from)), Vector2::Distance(from, to));
    }
    Vector2 &operator[](int i)
    {
        switch (i)
        {
        case 0:
            return from;
            break;
        case 1:
            return to;
            break;
        default:
            throw "数组越界";
            break;
        }
    }
};
```

# 二维多边形

```cpp
struct Polygon2
{
    std::vector<Vector2> points;

private:
    Vector2 accordance;

public:
    Polygon2 ConvexHull()
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
                // if (stk.size() >= 2)
                // {
                //     auto c = Vector2::Cross(stk.back() - stk[stk.size() - 2], i - stk.back());
                //     cerr << "c:" << c << endl;
                // }
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

    /*凸多边形用逆时针排序*/
    void autoanticlockwiselize()
    {
        accordance = average();
        anticlockwiselize();
    }

    // typedef bool(Polygon2::*comparator);

    void anticlockwiselize()
    {
        // comparator cmp = &Polygon2::anticlock_comparator;
        auto anticlock_comparator = [&](Vector2 &a, Vector2 &b) -> bool {
            return (a - accordance).toPolarCoordinate(false).y < (b - accordance).toPolarCoordinate(false).y;
        };
        std::sort(points.begin(), points.end(), anticlock_comparator);
        // for (auto i : points)
        // {
        //     cerr << (i - accordance).toPolarCoordinate() << "\t" << i << endl;
        // }
    }

    Vector2 average()
    {
        Vector2 avg(0, 0);
        for (auto i : points)
        {
            avg += i;
        }
        return avg / points.size();
    }

    /*求周长*/
    FLOAT_ perimeter()
    {
        FLOAT_ ret = Vector2::Distance(points.front(), points.back());
        for (int i = 1; i < points.size(); i++)
            ret += Vector2::Distance(points[i], points[i - 1]);
        return ret;
    }
    /*面积*/
    FLOAT_ area()
    {
        FLOAT_ ret = Vector2::Cross(points.back(), points.front());
        for (int i = 1; i < points.size(); i++)
            ret = ret + Vector2::Cross(points[i - 1], points[i]);
        return ret / 2;
    }
    /*求几何中心（形心、重心）*/
    Vector2 center()
    {
        Vector2 ret = (points.back() + points.front()) * Vector2::Cross(points.back(), points.front());
        for (int i = 1; i < points.size(); i++)
            ret = ret + (points[i - 1] + points[i]) * Vector2::Cross(points[i - 1], points[i]);
        return ret / area() / 6;
    }
    /*求边界整点数*/
    long long boundary_points()
    {
        long long b = 0;
        for (int i = 0; i < points.size() - 1; i++)
        {
            b += std::__gcd((long long)abs(points[i + 1].x - points[i].x), (long long)abs(points[i + 1].y - points[i].y));
        }
        return b;
    }
    /*Pick定理：多边形面积=内部整点数+边界上的整点数/2-1；求内部整点数*/
    long long interior_points(FLOAT_ A = -1, long long b = -1)
    {
        if (A < 0)
            A = area();
        if (b < 0)
            b = boundary_points();
        return (long long)A + 1 - (b / 2);
    }
};
```