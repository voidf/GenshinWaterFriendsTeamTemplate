#include "Headers.cpp"

namespace BalancedTree
{
/* 指定宏use_ptr使用指针定位左右儿子，指针可能会被搬家表传统艺能影响导致找不到地址 */
#ifdef use_ptr
// using P = Node<T> *;
#define P Node<T> *
#else
	using P = int;
#endif
	template <class T>
	struct Node
	{
		T v, su;
		int siz;
		bool rev;
		P f;
		P son[2];
		Node() {}
	};
	inline int mid(int l, int r) { return l + r >> 1; }

	template <class T>
	struct Splay
	{
		int siz;
		using ND = Node<T>;
		std::vector<ND> D;
		std::vector<int> gc; // 删除节点垃圾收集

		P root;
		P NIL; // 0号点是保留的哨兵点
		inline ND &resolve(P x)
		{
#ifdef use_ptr
			return *x;
#else
			return D[x];
#endif
		}
		inline P getref(ND &x)
		{
#ifdef use_ptr
			return &x;
#else
			return &x - &D.front();
#endif
		}
		inline ND &father(ND &x) { return resolve(x.f); }
		inline ND &lson(ND &x) { return resolve(x.son[0]); }
		inline ND &rson(ND &x) { return resolve(x.son[1]); }
		inline void pushup(ND &x)
		{
			x.siz = 1 + lson(x).siz + rson(x).siz;
			x.su = lson(x).su ^ rson(x).su ^ x.v;
		}
		inline void pinrev(ND &x)
		{
			x.rev ^= 1;
		}

		inline void pushdown(ND &x)
		{
			if (x.rev)
			{
				std::swap(x.son[0], x.son[1]);
				if (x.son[0] != NIL)
					pinrev(lson(x));
				if (x.son[1] != NIL)
					pinrev(rson(x));
				x.rev = 0;
			}
		}
		Splay(int size)
		{
			D.reserve(size + 1);
			gc.reserve(size);
			D.emplace_back();
			D[0].siz = D[0].rev = 0;
			D[0].f = D[0].son[0] = D[0].son[1] = getref(D[0]);
			root = NIL = getref(D[0]);
			siz = 0;
		}
		inline ND &allocate(T val, P father)
		{
			++siz;
			if (gc.size())
			{
				ND &b = D[gc.back()];
				b.su = b.v = val;
				b.siz = 1;
				b.f = father;
				b.rev = 0;
				b.son[0] = b.son[1] = NIL;
				gc.pop_back();
				return b;
			}
			else
			{
				D.emplace_back();
				ND &b = D.back();
				b.su = b.v = val;
				b.siz = 1;
				b.f = father;
				b.rev = 0;
				b.son[0] = b.son[1] = NIL;

				return b;
			}
		}

		inline void rotate(ND &x)
		{
			ND &y = resolve(x.f);
			ND &z = resolve(y.f);
			bool k = getref(x) == y.son[1];
			z.son[z.son[1] == getref(y)] = getref(x);
			x.f = getref(z);
			y.son[k] = x.son[!k];
			resolve(x.son[!k]).f = getref(y);
			x.son[!k] = getref(y);
			y.f = getref(x);
			pushup(y);
			pushup(x);
		}
		/*将x旋为goal的儿子 */
		inline void splay(ND &x, ND &goal)
		{
			while (x.f != getref(goal))
			{
				ND &y = resolve(x.f);
				ND &z = resolve(y.f);
				if (getref(z) != getref(goal))
					(z.son[1] == getref(y)) ^ (y.son[1] == getref(x)) ? rotate(x) : rotate(y);
				rotate(x);
			}
			if (getref(goal) == NIL)
				root = getref(x);
		}

		T *arr;
		P _build(int l, int r, P fa)
		{
			if (l > r)
				return NIL;
			int m = mid(l, r);
			ND &C = allocate(arr[m], fa);
			C.son[0] = _build(l, m - 1, getref(C));
			C.son[1] = _build(m + 1, r, getref(C));
			pushup(C);
			return getref(C);
		}

		void build(T *_arr, int siz, int beginwith = 0)
		{
			arr = _arr;
			// siz = _siz;
			root = _build(beginwith, beginwith + siz - 1, NIL);
		}

		/* insert在维护区间reverse以后就不能用，意义不一样 */
		inline void insert(const T x)
		{
			P u = root;
			P ff = NIL;
			while (u != NIL)
			{
				ff = u;
				u = resolve(u).son[resolve(u).v < x];
			}
			ND &U = allocate(x, ff);
			u = getref(U);
			if (ff != NIL)
			{
				resolve(ff).son[resolve(ff).v < x] = u;
			}
			splay(U, D[0]);
			// ++siz;
		}
		/* 从0开始 */
		inline ND &kth(int k)
		{
			P u = root;
			while (1)
			{
				ND &U = resolve(u);
				pushdown(U);
				ND &ls = lson(U);
				if (ls.siz > k)
					u = U.son[0];
				else if (ls.siz == k)
					return U;
				else
					k -= ls.siz + 1, u = U.son[1];
			}
		}

		inline void reverse(int l, int r)
		{
			if (l <= 0 and r >= siz - 1)
			{
				pinrev(resolve(root));
			}
			else if (l <= 0)
			{
				splay(kth(r + 1), D[0]);
				pinrev(lson(resolve(root)));
			}
			else if (r >= siz - 1)
			{
				splay(kth(l - 1), D[0]);
				pinrev(rson(resolve(root)));
			}
			else
			{
				ND &L = kth(l - 1);
				ND &R = kth(r + 1);
				splay(L, D[0]);
				splay(R, L);
				pinrev(lson(rson(resolve(root))));
			}
		}

		std::function<void(T)> tempf;
		void _foreach(ND &x)
		{
			pushdown(x);
			if (x.son[0] != NIL)
				_foreach(resolve(x.son[0]));
			if (getref(x) != NIL)
				tempf(x.v);
			if (x.son[1] != NIL)
				_foreach(resolve(x.son[1]));
		}
		void foreach (std::function<void(T)> F)
		{
			tempf = F;
			_foreach(resolve(root));
		}
	};

	template <typename T>
	struct LCT : public Splay<T>
	{
		// using Splay<T>::ND;
		using ND = Node<T>;
		using Splay<T>::getref;
		using Splay<T>::resolve;
		using Splay<T>::rson;
		using Splay<T>::lson;
		using Splay<T>::father;
		using Splay<T>::pushup;
		using Splay<T>::pinrev;
		using Splay<T>::pushdown;
		using Splay<T>::NIL;
		LCT(int size) : Splay<T>(size) {}

		inline bool isnot_root(ND &x)
		{
			return getref(lson(father(x))) == getref(x) or getref(rson(father(x))) == getref(x);
		}

		inline void rotate(ND &x)
		{
			ND &y = father(x);
			ND &z = father(y);
			bool k = getref(x) == y.son[1];
			P rw = x.son[!k];
			if (isnot_root(y))
				z.son[z.son[1] == getref(y)] = getref(x);
			x.son[!k] = getref(y);
			y.son[k] = rw;
			if (rw != NIL)
				resolve(rw).f = getref(y);
			y.f = getref(x);
			x.f = getref(z);
			pushup(y);
		}

		inline void splay(ND &x)
		{
			P ry = getref(x);
			vector<P> stk(1, ry);
			while (isnot_root(resolve(ry)))
				stk.emplace_back(ry = resolve(ry).f);
			while (stk.size())
			{
				pushdown(resolve(stk.back()));
				stk.pop_back();
			}
			while (isnot_root(x))
			{
				ry = x.f;
				ND &y = resolve(ry);
				ND &z = resolve(y.f);
				if (isnot_root(y))
					rotate((y.son[0] == getref(x)) ^ (z.son[0] == ry) ? x : y);
				rotate(x);
			}
			pushup(x);
		}

		inline void access(ND &x)
		{
			P rx = getref(x);
			for (P ry = NIL; rx != NIL; rx = resolve(ry = rx).f)
				splay(resolve(rx)), resolve(rx).son[1] = ry, pushup(resolve(rx));
		}

		inline void chroot(ND &x)
		{
			access(x);
			splay(x);
			pinrev(x);
		}

		inline ND &findroot(ND &x)
		{
			access(x);
			splay(x);
			P rx = getref(x);
			while (resolve(rx).son[0] != NIL)
				pushdown(resolve(rx)), rx = resolve(rx).son[0];
			splay(resolve(rx));
			return resolve(rx);
		}
		/* 路径分离出来之后y上的su值即为x->y上路径的信息() */
		inline void split(ND &x, ND &y)
		{
			chroot(x);
			access(y);
			splay(y);
		}
		inline bool link(ND &x, ND &y)
		{
			chroot(x);
			if (getref(findroot(y)) != getref(x))
			{
				x.f = getref(y);
				return true;
			}
			return false;
		}
		inline bool cut(ND &x, ND &y)
		{
			chroot(x);
			if (getref(findroot(y)) == getref(x) and y.f == getref(x) and y.son[0] == NIL)
			{
				y.f = x.son[1] = NIL;
				pushup(x);
				return true;
			}
			return false;
		}
	};

};
