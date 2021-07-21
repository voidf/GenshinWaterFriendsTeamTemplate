#include "Geo_Base.cpp"
#include "Geo_Vector3.cpp"

namespace Geometry
{
    /* https://www.luogu.com.cn/record/51674409 模板题需要用long double */
    
    struct Sphere
	{
		FLOAT_ radius;
		Vector3 center;
		Sphere(Vector3 c, FLOAT_ r) : center(c), radius(r) {}
		Sphere(FLOAT_ x, FLOAT_ y, FLOAT_ z, FLOAT_ r) : center(x, y, z), radius(r) {}
		FLOAT_ volumn() { return 4.0 * PI * pow(radius, 3) / 3.0; }
		FLOAT_ intersectVolumn(Sphere o)
		{
			Vector3 dist = o.center - center;
			FLOAT_ distval = dist.magnitude();
			if (distval > o.radius + radius)
				return 0;
			if (distval < abs(o.radius - radius))
			{
				return o.radius > radius ? volumn() : o.volumn();
			}
			FLOAT_ &d = distval;
			//球心距
			FLOAT_ t = (d * d + o.radius * o.radius - radius * radius) / (2.0 * d);
			//h1=h2，球冠的高
			FLOAT_ h = sqrt((o.radius * o.radius) - (t * t)) * 2;
			FLOAT_ angle_a = 2 * acos((o.radius * o.radius + d * d - radius * radius) / (2.0 * o.radius * d)); //余弦公式计算r1对应圆心角，弧度
			FLOAT_ angle_b = 2 * acos((radius * radius + d * d - o.radius * o.radius) / (2.0 * radius * d));   //余弦公式计算r2对应圆心角，弧度
			FLOAT_ l1 = ((o.radius * o.radius - radius * radius) / d + d) / 2;
			FLOAT_ l2 = d - l1;
			FLOAT_ x1 = o.radius - l1, x2 = radius - l2;	//分别为两个球缺的高度
			FLOAT_ v1 = PI * x1 * x1 * (o.radius - x1 / 3); //相交部分r1圆所对应的球缺部分体积
			FLOAT_ v2 = PI * x2 * x2 * (radius - x2 / 3);	//相交部分r2圆所对应的球缺部分体积
															//相交部分体积
			return v1 + v2;
		}
		FLOAT_ joinVolumn(Sphere o)
		{
			return volumn() + o.volumn() - intersectVolumn(o);
		}
	};



}