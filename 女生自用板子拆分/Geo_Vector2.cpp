#ifndef Geo_vec2_H
#define Geo_vec2_H

#include "Geo_Base.cpp"

namespace Geometry
{

	struct vec2
	{
		fl x, y;
		vec2(fl _x, fl _y) : x(_x), y(_y) {}
		vec2(fl n) : x(n), y(n) {}
		// vec2(const glm::vec2& v) : x(v.x), y(v.y) {}
		// inline glm::vec2 toglm(){return {x, y};}
		vec2() : x(0.0), y(0.0) {}
		inline vec2 &operator=(const vec2 &b)
		{
			this->x = b.x;
			this->y = b.y;
			return *this;
		}
		inline bool operator==(const vec2 &b) const { return round_compare(this->x, b.x) and round_compare(this->y, b.y); }
		inline bool operator!=(const vec2 &b) const { return not((*this) == b); }
		inline fl &operator[](const int ind)
		{
			switch (ind)
			{
			case 0:
				return (this->x);
				break;
			case 1:
				return (this->y);
				break;
			case 'x':
				return (this->x);
				break;
			case 'y':
				return (this->y);
				break;
			default:
				throw "无法理解除0,1外的索引";
				break;
			}
		}
		/* 转为字符串 */
		inline std::string ToString() const
		{
			std::ostringstream ostr;
			ostr << "vec2(" << this->x << ", " << this->y << ")";
			return ostr.str();
		}
		inline vec2 operator-() const { return vec2(-x, -y); }

		inline friend std::ostream &operator<<(std::ostream &o, const vec2 &v) { return o << v.ToString(); }
		inline vec2 &operator+=(const vec2 &b)
		{
			x += b.x, y += b.y;
			return (*this);
		}
		inline vec2 &operator-=(const vec2 &b)
		{
			x -= b.x, y -= b.y;
			return (*this);
		}
		inline vec2 &operator*=(const vec2 &b)
		{
			x *= b.x, y *= b.y;
			return (*this);
		}
		inline vec2 &operator/=(const vec2 &b)
		{
			x /= b.x, y /= b.y;
			return (*this);
		}
		inline vec2 &operator+=(const fl &n)
		{
			x += n, y += n;
			return (*this);
		}
		inline vec2 &operator-=(const fl &n)
		{
			x -= n, y -= n;
			return (*this);
		}
		inline vec2 &operator*=(const fl &n)
		{
			x *= n, y *= n;
			return (*this);
		}
		inline vec2 &operator/=(const fl &n)
		{
			x /= n, y /= n;
			return (*this);
		}

		inline vec2 operator+(const vec2 &b) const { return vec2(*this) += b; }
		inline vec2 operator-(const vec2 &b) const { return vec2(*this) -= b; }
		inline vec2 operator*(const vec2 &b) const { return vec2(*this) *= b; }
		inline vec2 operator/(const vec2 &b) const { return vec2(*this) /= b; }
		inline vec2 operator+(const fl &n) const { return vec2(*this) += n; }
		inline vec2 operator-(const fl &n) const { return vec2(*this) -= n; }
		inline vec2 operator*(const fl &n) const { return vec2(*this) *= n; }
		inline vec2 operator/(const fl &n) const { return vec2(*this) /= n; }
		inline friend vec2 operator+(const fl &n, const vec2 &b) { return vec2(n) += b; }
		inline friend vec2 operator-(const fl &n, const vec2 &b) { return vec2(n) -= b; }
		inline friend vec2 operator*(const fl &n, const vec2 &b) { return vec2(n) *= b; }
		inline friend vec2 operator/(const fl &n, const vec2 &b) { return vec2(n) /= b; }

		/* 绕原点逆时针旋转多少度 */
		inline void rotate(fl theta, bool use_degree = false)
		{
			fl ox = x;
			fl oy = y;
			theta = (use_degree ? theta / 180 * PI : theta);
			fl costheta = cos(theta);
			fl sintheta = sin(theta);
			this->x = ox * costheta - oy * sintheta;
			this->y = oy * costheta + ox * sintheta;
		}

		inline bool operator<(const vec2 &b) const { return this->x < b.x or this->x == b.x and this->y < b.y; }

		/* 向量的平方模 */
		inline fl sqrMagnitude() const { return x * x + y * y; }
		inline fl mag2() const { return sqrMagnitude(); }
		/* 向量的模 */
		inline fl magnitude() const { return sqrt(this->sqrMagnitude()); }
		inline fl mag() const { return magnitude(); }

		/* 用极坐标换算笛卡尔坐标 */
		inline static vec2 fromPolarCoordinate(const vec2 &v, bool use_degree = 1) { return v.toCartesianCoordinate(use_degree); }

		/* 转为笛卡尔坐标 */
		inline vec2 toCartesianCoordinate(bool use_degree = 1) const
		{
			return vec2(
				x * cos(y * (use_degree ? PI / 180.0 : 1)),
				x * sin(y * (use_degree ? PI / 180.0 : 1)));
		}
		/* 转为极坐标 */
		inline vec2 toPolarCoordinate(bool use_degree = 1) const
		{
			return vec2(
				magnitude(),
				toPolarAngle(use_degree));
		}

		/* 获取极角 */
		inline fl toPolarAngle(bool use_degree = 1) const
		{
			fl ret = atan2(y, x);
			if (ret < 0)
				ret += PI * 2;
			return ret * (use_degree ? 180.0 / PI : 1);
		}
		/* 转为极坐标 */
		inline static vec2 ToPolarCoordinate(const vec2 &coordinate, bool use_degree = 1) { return coordinate.toPolarCoordinate(use_degree); }

		/* 向量单位化 */
		inline void Normalize()
		{
			fl _m = this->magnitude();
			this->x /= _m;
			this->y /= _m;
		}

		/* 返回与该向量方向同向的单位向量 */
		inline vec2 normalized() const
		{
			fl _m = this->magnitude();
			return vec2(this->x / _m, this->y / _m);
		}
		/* 距离 */
		inline static fl Distance(const vec2 &a, const vec2 &b) { return (a - b).magnitude(); }
		inline fl dis(const vec2 &b) { return Distance(*this, b); }
		/* 平方距离 */
		inline static fl sqrDistance(const vec2 &a, const vec2 &b) { return (a - b).sqrMagnitude(); }
		inline fl dis2(const vec2 &b) { return sqrDistance(*this, b); }

		/* 向量线性插值 */
		inline static vec2 lerp(const vec2 &a, const vec2 &b, const fl &t) { return a + (b - a) * t; }

		/* 向量圆形插值，不可靠 */
		inline static vec2 SlerpUnclamped(vec2 a, vec2 b, const fl &t)
		{
			auto si = SignedRad(a, b);
			a.rotate(t * si);
			return a;
			// a = a.toPolarCoordinate();
			// b = b.toPolarCoordinate();
			// return lerp(a, b, t).toCartesianCoordinate();
		}

		/* 拿它的垂直向量（逆时针旋转90°） */
		inline static vec2 Perpendicular(const vec2 &inDirection) { return vec2(-inDirection.y, inDirection.x); }
		/* 根据inNormal法向反射inDirection向量，参考光的平面镜反射，入射光为inDirection，平面镜的法线为inNormal */
		inline static vec2 Reflect(const vec2 &inDirection, const vec2 &inNormal) { return inDirection - 2 * vec2::Dot(inDirection, inNormal) * inNormal; }
		/* 点积 */
		inline static fl Dot(const vec2 &lhs, const vec2 &rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
		inline fl dot(const vec2 &b) const { return Dot(*this, b); }
		/* 叉积 */
		inline static fl Cross(const vec2 &lhs, const vec2 &rhs) { return lhs.x * rhs.y - lhs.y * rhs.x; }
		inline fl cross(const vec2 &b) const { return Cross(*this, b); }

		/* 有符号弧度夹角 */
		inline static fl SignedRad(const vec2 &from, const vec2 &to) { return atan2(vec2::Cross(from, to), vec2::Dot(from, to)); }
		/* 无符号弧度夹角 */
		inline static fl Rad(const vec2 &from, const vec2 &to) { return abs(vec2::SignedRad(from, to)); }
		/* 有符号角度夹角 */
		inline static fl SignedAngle(const vec2 &from, const vec2 &to) { return vec2::SignedRad(from, to) * 180.0 / PI; }
		/* 无符号角度夹角 */
		inline static fl Angle(const vec2 &from, const vec2 &to) { return abs(vec2::SignedAngle(from, to)); }

		/* 返回俩向量中x的最大值和y的最大值构造而成的向量 */
		inline static vec2 Max(const vec2 &lhs, const vec2 &rhs) { return vec2(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y)); }

		/* 返回俩向量中x的最小值和y的最小值构造而成的向量 */
		inline static vec2 Min(const vec2 &lhs, const vec2 &rhs) { return vec2(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y)); }

		/* 获得vector在onNormal方向的投影，无损，无需单位化写法 */
		inline static vec2 Project(const vec2 &vector, const vec2 &onNormal) { return Dot(vector, onNormal) / onNormal.sqrMagnitude() * onNormal; }

		inline static fl ProjectLength(const vec2 &vector, const vec2 &onNormal) { return Project(vector, onNormal).magnitude(); }

		/* 判断p是否在向量from->to的延长线上，精度不高，慎用 */
		inline static bool indirection(const vec2 &from, const vec2 &to, const vec2 &p)
		{
			vec2 p1 = to - from;
			vec2 p2 = p - from;
			if (!intereps(Cross(p1, p2)) || Dot(p1, p2) <= 0)
				return false;
			return (p1.sqrMagnitude() < p2.sqrMagnitude());
		}

		/* 判断p是否在线段[from -> to]上，精度不高，慎用 */
		inline static bool inrange(const vec2 &from, const vec2 &to, const vec2 &p)
		{
			if (p == from || p == to)
				return true;
			vec2 p1 = to - from;
			vec2 p2 = p - from;
			if (!intereps(Cross(p1, p2)) || Dot(p1, p2) <= 0)
				return false;
			return (p1.sqrMagnitude() >= p2.sqrMagnitude());
		}

		/* 判断三个点是否共线 */
		inline static bool Collinear(const vec2 &a, const vec2 &b, const vec2 &c)
		{
			return round_compare(Cross(c - a, b - a), 0.0);
		}

		using itr = std::vector<vec2>::iterator;
		static void solve_nearest_pair(const itr l, const itr r, fl &ans)
		{
			if (r - l <= 1)
				return;
			std::vector<itr> Q;
			itr t = l + (r - l) / 2;
			fl w = t->x;
			solve_nearest_pair(l, t, ans), solve_nearest_pair(t, r, ans);
			std::inplace_merge(l, t, r, [](const vec2 &a, const vec2 &b) -> bool
							   { return a.y < b.y; });
			for (itr x = l; x != r; ++x)
				if ((w - x->x) * (w - x->x) <= ans)
					Q.emplace_back(x);
			for (auto x = Q.begin(), y = x; x != Q.end(); ++x)
			{
				while (y != Q.end() && pow((*y)->y - (*x)->y, 2) <= ans)
					++y;
				for (auto z = x + 1; z != y; ++z)
					ans = min(ans, (**x - **z).sqrMagnitude());
			}
		}
		/* 平面最近点对 入口 */
		inline static fl nearest_pair(std::vector<vec2> &V)
		{
			sort(V.begin(), V.end(), [](const vec2 &a, const vec2 &b) -> bool
				 { return a.x < b.x; });
			fl ans = (V[0] - V[1]).sqrMagnitude();
			solve_nearest_pair(V.begin(), V.end(), ans);
			return ans;
		}
	};

	struct PolarSortCmp
	{
		inline bool operator()(const vec2 &a, const vec2 &b) const { return a.toPolarAngle(0) < b.toPolarAngle(0); }
	};
	/* 相等的向量可能不会贴着放，不能保证排完之后遍历一圈是旋转360°，慎用 */
	struct CrossSortCmp
	{
		inline bool operator()(const vec2 &a, const vec2 &b) const { return vec2::Cross(a, b) > 0; }
	};

}

#endif