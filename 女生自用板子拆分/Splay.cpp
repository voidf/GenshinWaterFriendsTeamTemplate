#include "Headers.cpp"

// #define use_ptr
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
		T v;
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
		P NIL;
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
			if (gc.size())
			{
				ND &b = D[gc.back()];
				b.v = val;
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
				b.v = val;
				b.f = father;
				b.rev = 0;
				b.son[0] = b.son[1] = NIL;
				return b;
			}
		}
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

		inline ND &lson(ND &x) { return resolve(x.son[0]); }
		inline ND &rson(ND &x) { return resolve(x.son[1]); }
		inline void pushup(ND &x) { x.siz = 1 + lson(x).siz + rson(x).siz; }
		inline void pushdown(ND &x)
		{
			if (x.rev)
			{
				rson(x).rev ^= 1;
				lson(x).rev ^= 1;
				x.rev = 0;
				std::swap(x.son[0], x.son[1]);
			}
		}
		inline void rotate(ND &x)
		{
			// if (getref(x) == NIL)
			// return;
			ND &y = resolve(x.f);
			ND &z = resolve(y.f);
			bool k = getref(x) == y.son[1];
			// if (getref(z) != NIL)
			z.son[z.son[1] == getref(y)] = getref(x);
			x.f = getref(z);
			// if (getref(y) != NIL)
			y.son[k] = x.son[k ^ 1];
			resolve(x.son[k ^ 1]).f = getref(y);
			x.son[k ^ 1] = getref(y);
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
		inline void insert(const T x)
		{
			P u = root;
			P ff = NIL;
			while (u != NIL)
			{
				ff = u;
				u = resolve(u).son[x > resolve(u).v];
			}

			// D.emplace_back(D[0]);
			ND &U = allocate(x, ff);
			u = getref(U);
			if (ff != NIL)
			{
				resolve(ff).son[x > resolve(ff).v] = u;
			}
			// U.init(x, ff, NIL);
			// U.f = ff;
			// U.v = x;
			splay(U, D[0]);
			++siz;
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
				resolve(root).rev ^= 1;
			}
			else if (l <= 0)
			{
				splay(kth(r + 1), D[0]);
				lson(resolve(root)).rev ^= 1;
			}
			else if (r >= siz - 1)
			{
				splay(kth(l - 1), D[0]);
				rson(resolve(root)).rev ^= 1;
			}
			else
			{
				ND &L = kth(l - 1);
				ND &R = kth(r + 1);
				splay(L, D[0]);
				splay(R, L);
				lson(rson(resolve(root))).rev ^= 1;
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
};
