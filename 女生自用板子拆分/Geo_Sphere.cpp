#ifndef Geo_Sphere_H
#define Geo_Sphere_H

#include "Geo_Base.cpp"
#include "Geo_vec3.cpp"

namespace Geometry
{
	/* https://www.luogu.com.cn/record/51674409 模板题需要用long double */

	struct Sphere
	{
		fl radius;
		vec3 center;
		Sphere(vec3 c, fl r) : center(c), radius(r) {}
		Sphere(fl x, fl y, fl z, fl r) : center(x, y, z), radius(r) {}
		fl volumn() const { return 4.0 * PI * pow(radius, 3) / 3.0; }
		fl intersectVolumn(const Sphere &o) const
		{
			vec3 dist = o.center - center;
			fl distval = dist.magnitude();
			if (distval > o.radius + radius)
				return 0;
			if (distval < abs(o.radius - radius))
			{
				return o.radius > radius ? volumn() : o.volumn();
			}
			fl &d = distval;
			//球心距
			fl t = (d * d + o.radius * o.radius - radius * radius) / (2.0 * d);
			//h1=h2，球冠的高
			fl h = sqrt((o.radius * o.radius) - (t * t)) * 2;
			fl angle_a = 2 * acos((o.radius * o.radius + d * d - radius * radius) / (2.0 * o.radius * d)); //余弦公式计算r1对应圆心角，弧度
			fl angle_b = 2 * acos((radius * radius + d * d - o.radius * o.radius) / (2.0 * radius * d));   //余弦公式计算r2对应圆心角，弧度
			fl l1 = ((o.radius * o.radius - radius * radius) / d + d) / 2;
			fl l2 = d - l1;
			fl x1 = o.radius - l1, x2 = radius - l2;	//分别为两个球缺的高度
			fl v1 = PI * x1 * x1 * (o.radius - x1 / 3); //相交部分r1圆所对应的球缺部分体积
			fl v2 = PI * x2 * x2 * (radius - x2 / 3);	//相交部分r2圆所对应的球缺部分体积
														//相交部分体积
			return v1 + v2;
		}
		fl joinVolumn(const Sphere &o) const
		{
			return volumn() + o.volumn() - intersectVolumn(o);
		}
		/* 直线与圆的交点，求出参数t，交点即为s[0] + s.toward() * t */
		std::pair<fl, fl> intersectT(const Segment3 &s) const
		{
			vec3 tw = s.toward();
			vec3 dt = s.at(0) - center;
			fl a = tw.mag2();
			fl b = dt.dot(tw) * fl(2);
			fl c = dt.mag2() - radius * radius;
			return solveQuadraticEquation(a, b, c);
		}
	};

}
#endif
