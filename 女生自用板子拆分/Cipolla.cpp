#include "Headers.cpp"
#include "MathTheoryMisc.cpp"

/* def94200d616892a0187be01c94ea9c1 使用Cipolla计算二次剩余 */
template <typename T>
struct Cipolla
{
	T re_al, im_ag;
	/* 定义I = a^2 - n，实际上是单位负根的平方 */
	inline static T mod, I; // 17特性，不行就提全局

	inline static Cipolla power(Cipolla x, LL p)
	{
		Cipolla res(1);
		while (p)
		{
			if (p & 1)
				res = res * x;
			x = x * x;
			p >>= 1;
		}
		return res;
	}
	/* 检查x是不是二次剩余 */
	inline static bool check_if_residue(T x)
	{
		return power(x, mod - 1 >> 1) == 1;
	}

	/* 算法入口，要求p是奇素数 */
	static void solve(T n, T p, T &x0, T &x1)
	{
		n %= p;
		mod = p;
		if (n == 0)
		{
			x0 = x1 = 0;
			return;
		}
		if (!check_if_residue(n))
		{
			x0 = x1 = -1; // 无解
			return;
		}
		T a;
		do
		{
			a = randint(T(1), mod - 1);
		} while (check_if_residue((a * a + mod - n) % mod));
		I = (a * a - n + mod) % mod;
		x0 = T(power(Cipolla(a, 1), mod + 1 >> 1).real());
		x1 = mod - x0;
	}
	/* 实际上是个模意义复数类 */
	Cipolla(T _real = 0, T _imag = 0) : re_al(_real), im_ag(_imag) {}
	inline T &real() { return re_al; }
	inline T &imag() { return im_ag; }
	inline bool operator==(const Cipolla &y) const
	{
		return re_al == y.re_al and im_ag == y.im_ag;
	}
	inline Cipolla operator*(const Cipolla &y) const
	{
		return Cipolla((re_al * y.re_al + I * im_ag % mod * y.im_ag) % mod,
					   (im_ag * y.re_al + re_al * y.im_ag) % mod);
	}
};