#include "Geo_Base.cpp"

namespace Geometry
{

	struct Vector2
	{
		FLOAT_ x, y;
		Vector2(FLOAT_ _x, FLOAT_ _y) : x(_x), y(_y) {}
		Vector2(FLOAT_ n) : x(n), y(n) {}
		Vector2() : x(0.0), y(0.0) {}
		inline Vector2 &operator=(const Vector2 &b)
		{
			this->x = b.x;
			this->y = b.y;
			return *this;
		}
		inline bool operator==(const Vector2 &b) const { return round_compare(this->x, b.x) and round_compare(this->y, b.y); }
		inline bool operator!=(const Vector2 &b) const { return not((*this) == b); }
		inline FLOAT_ &operator[](const int ind) const
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
		inline friend std::ostream &operator<<(std::ostream &o, const Vector2 &v) const { return o << v.ToString(); }
		inline Vector2 &operator+=(const Vector2 &b)
		{
			x += b.x, y += b.y;
			return (*this);
		}
		inline Vector2 &operator-=(const Vector2 &b)
		{
			x -= b.x, y -= b.y;
			return (*this);
		}
		inline Vector2 &operator*=(const Vector2 &b)
		{
			x *= b.x, y *= b.y;
			return (*this);
		}
		inline Vector2 &operator/=(const Vector2 &b)
		{
			x /= b.x, y /= b.y;
			return (*this);
		}
		inline Vector2 &operator+=(const FLOAT_ &n)
		{
			x += n, y += n;
			return (*this);
		}
		inline Vector2 &operator-=(const FLOAT_ &n)
		{
			x -= n, y -= n;
			return (*this);
		}
		inline Vector2 &operator*=(const FLOAT_ &n)
		{
			x *= n, y *= n;
			return (*this);
		}
		inline Vector2 &operator/=(const FLOAT_ &n)
		{
			x /= n, y /= n;
			return (*this);
		}
		inline Vector2 operator+(const Vector2 &b) const { return Vector2(*this) + b; }
		inline Vector2 operator-(const Vector2 &b) const { return Vector2(*this) - b; }
		inline Vector2 operator*(const Vector2 &b) const { return Vector2(*this) * b; }
		inline Vector2 operator/(const Vector2 &b) const { return Vector2(*this) / b; }
		inline Vector2 operator+(const FLOAT_ &n) const { return Vector2(*this) + n; }
		inline Vector2 operator-(const FLOAT_ &n) const { return Vector2(*this) - n; }
		inline Vector2 operator*(const FLOAT_ &n) const { return Vector2(*this) * n; }
		inline Vector2 operator/(const FLOAT_ &n) const { return Vector2(*this) / n; }
		inline friend Vector2 operator+(const FLOAT_ &n, const Vector2 &b) { return Vector2(n) + b; }
		inline friend Vector2 operator-(const FLOAT_ &n, const Vector2 &b) { return Vector2(n) - b; }
		inline friend Vector2 operator*(const FLOAT_ &n, const Vector2 &b) { return Vector2(n) * b; }
		inline friend Vector2 operator/(const FLOAT_ &n, const Vector2 &b) { return Vector2(n) / b; }

		/* 绕原点逆时针旋转多少度 */
		inline void rotate(const FLOAT_ &theta, bool use_degree = false)
		{
			FLOAT_ ox = x;
			FLOAT_ oy = y;
			theta = (use_degree ? theta / 180 * PI : theta);
			FLOAT_ costheta = cos(theta);
			FLOAT_ sintheta = sin(theta);
			this->x = ox * costheta - oy * sintheta;
			this->y = oy * costheta + ox * sintheta;
		}

		inline bool operator<(const Vector2 &b) const { return this->x < b.x or this->x == b.x and this->y < b.y; }

		/* 向量的平方模 */
		inline FLOAT_ sqrMagnitude() const { return pow(this->x, 2) + pow(this->y, 2); }
		/* 向量的模 */
		inline FLOAT_ magnitude() const { return sqrt(this->sqrMagnitude()); }
		/*判等*/
		inline bool equals(const Vector2 &b) { return (*this) == b; }

		/*用极坐标换算笛卡尔坐标*/
		inline static Vector2 fromPolarCoordinate(const Vector2 &v, bool use_degree = 1) { return v.toCartesianCoordinate(use_degree); }

		/*转为笛卡尔坐标*/
		inline Vector2 toCartesianCoordinate(bool use_degree = 1) const
		{
			return Vector2(
				x * cos(y * (use_degree ? PI / 180.0 : 1)),
				x * sin(y * (use_degree ? PI / 180.0 : 1)));
		}
		/*转为极坐标*/
		inline Vector2 toPolarCoordinate(bool use_degree = 1) const
		{
			return Vector2(
				magnitude(),
				toPolarAngle(use_degree));
		}

		/*获取极角*/
		inline FLOAT_ toPolarAngle(bool use_degree = 1) const { return atan2(y, x) * (use_degree ? 180.0 / PI : 1); }

		/*转为极坐标*/
		inline static Vector2 ToPolarCoordinate(const Vector2 &coordinate, bool use_degree = 1) { return coordinate.toPolarCoordinate(use_degree); }

		/* 向量单位化 */
		inline void Normalize()
		{
			FLOAT_ _m = this->magnitude();
			this->x /= _m;
			this->y /= _m;
		}

		/*转为字符串*/
		inline std::string ToString() const
		{
			std::ostringstream ostr;
			ostr << "Vector2(" << this->x << ", " << this->y << ")";
			return ostr.str();
		}

		/* 返回与该向量方向同向的单位向量 */
		inline Vector2 normalized() const
		{
			FLOAT_ _m = this->magnitude();
			return Vector2(this->x / _m, this->y / _m);
		}
		// FLOAT_ Distance(Vector2 b) { return ((*this) - b).magnitude(); }
		/* 距离 */
		inline static FLOAT_ Distance(const Vector2 &a, const Vector2 &b) { return (a - b).magnitude(); }

		/*向量线性插值*/
		inline static Vector2 LerpUnclamped(const Vector2 &a, const Vector2 &b, const FLOAT_ &t) { return a + (b - a) * t; }

		/*向量圆形插值*/
		inline static Vector2 SlerpUnclamped(const Vector2 &a, const Vector2 &b, const FLOAT_ &t)
		{
			// Vector2 c = b - a;
			a = a.toPolarCoordinate();
			b = b.toPolarCoordinate();

			return LerpUnclamped(a, b, t).toCartesianCoordinate();
		}

		/* 拿它的垂直向量（逆时针旋转90°） */
		inline static Vector2 Perpendicular(const Vector2 &inDirection) const { return Vector2(-inDirection.y, inDirection.x); }
		/*根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal*/
		inline static Vector2 Reflect(const Vector2 &inDirection, const Vector2 &inNormal) { return inDirection - 2 * Vector2::Dot(inDirection, inNormal) * inNormal; }
		/* 点积 */
		inline static FLOAT_ Dot(const Vector2 &lhs, const Vector2 &rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
		/* 叉积 */
		inline static FLOAT_ Cross(const Vector2 &lhs, const Vector2 &rhs) { return lhs.x * rhs.y - lhs.y * rhs.x; }
		/*有符号弧度夹角*/
		inline static FLOAT_ SignedRad(const Vector2 &from, const Vector2 &to) { return atan2(Vector2::Cross(from, to), Vector2::Dot(from, to)); }
		/*无符号弧度夹角*/
		inline static FLOAT_ Rad(const Vector2 &from, const Vector2 &to) { return abs(Vector2::SignedRad(from, to)); }
		/*有符号角度夹角*/
		inline static FLOAT_ SignedAngle(const Vector2 &from, const Vector2 &to) { return Vector2::SignedRad(from, to) * 180.0 / PI; }
		/*无符号角度夹角*/
		inline static FLOAT_ Angle(const Vector2 &from, const Vector2 &to) { return abs(Vector2::SignedAngle(from, to)); }

		/*返回俩向量中x的最大值和y的最大值构造而成的向量*/
		inline static Vector2 Max(const Vector2 &lhs, const Vector2 &rhs) { return Vector2(max(lhs.x, rhs.x), max(lhs.y, rhs.y)); }

		/*返回俩向量中x的最小值和y的最小值构造而成的向量*/
		inline static Vector2 Min(const Vector2 &lhs, const Vector2 &rhs) { return Vector2(min(lhs.x, rhs.x), min(lhs.y, rhs.y)); }

		/*获得vector在onNormal方向的投影，onNormal需要单位化*/
		inline static Vector2 Project(const Vector2 &vector, const Vector2 &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }

		inline static FLOAT_ ProjectLength(const Vector2 &vector, const Vector2 &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude(); }
	};
	struct PolarSortCmp
	{
		inline bool operator()(const Vector2 &a, const Vector2 &b) const { return a.toPolarAngle(0) < b.toPolarAngle(0); }
	};
	/* 相等的向量可能不会贴着放，不能保证排完之后遍历一圈是旋转360°，慎用 */
	struct CrossSortCmp
	{
		inline bool operator()(const Vector2 &a, const Vector2 &b) const { return Vector2::Cross(a, b) > 0; }
	};

}