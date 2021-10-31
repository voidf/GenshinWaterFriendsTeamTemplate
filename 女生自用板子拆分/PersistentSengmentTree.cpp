
// #define use_ptr

namespace Persistent_seg
{
/* 指定宏use_ptr使用指针定位左右儿子，指针可能会被搬家表传统艺能影响导致找不到地址 */
#ifdef use_ptr
// using P = Node<T> *;
#define P Node<T> *
	P NIL = nullptr;
#else
	using P = int;
	P NIL = -1;
#endif
	template <class T>
	struct Node
	{
		T v, alz, mlz;
		P l = NIL;
		P r = NIL;
		Node() : v(0), alz(0), mlz(1) {}
		Node(T _v) : v(_v), alz(0), mlz(1) {}
	};
	inline int mid(int l, int r) { return l + r >> 1; }
	/* 用法:构造后用auto_reserve分配空间,然后build初始化,此时初始版本被填入H[0]中 */
	template <class T>
	struct PST_trad
	{
		int QL, QR;
		int LB, RB;
		using ND = Node<T>;
		std::vector<ND> D;
		std::vector<P> H;
		T *refarr;
		T TMP;
		bool new_version;
		ND &resolve(P x)
		{
#ifdef use_ptr
			return *x;
#else
			return D[x];
#endif
		}
		P getref(ND &x)
		{
#ifdef use_ptr
			return &x;
#else
			return &x - &D.front();
#endif
		}
		PST_trad() {}
		PST_trad(int n, int m) { auto_reserve(n, m); }
		void auto_reserve(int n, int m)
		{
			D.reserve((1 + ceil(log2(n))) * m + 2 * n);
			H.reserve(m);
		}

		void maintain(ND &x)
		{
			ND &lson = resolve(x.l);
			ND &rson = resolve(x.r);
			x.v = lson.v + rson.v;
		}

		void pushdown(ND &x, int l, int r)
		{
			ND &lson = resolve(x.l);
			ND &rson = resolve(x.r);
			if (x.mlz != 1)
			{
				lson.v *= x.alz;
				lson.alz *= x.mlz;
				lson.mlz *= x.mlz;
				rson.v *= x.alz;
				rson.alz *= x.mlz;
				rson.mlz *= x.mlz;
				x.mlz = 1;
			}
			if (x.alz != 0)
			{
				int m = mid(l, r);
				lson.v += x.alz * (m - l + 1);
				lson.alz += x.alz;
				rson.v += x.alz * (r - m);
				rson.alz += x.alz;
				x.alz = 0;
			}
		}

		P _build(int l, int r)
		{
			if (l == r)
			{
				if (refarr == nullptr)
					D.emplace_back();
				else
					D.emplace_back(*(refarr + l));
				return getref(D.back());
			}
			D.emplace_back();
			// ND &C = ;
			P rr = getref(D.back());
			int m = mid(l, r);
			resolve(rr).l = _build(l, m);
			resolve(rr).r = _build(m + 1, r);
			// cerr << "REF c:" << rr << endl;
			return rr;
		}
		/* 建默认空树可以给rf填nullptr */
		void build(T *rf, int l, int r)
		{
			refarr = rf;
			LB = l;
			RB = r;
			H.emplace_back(_build(l, r));
		}
		P _updatem(int l, int r, P o)
		{
			ND &old = resolve(o);
			if (new_version)
				D.emplace_back(old);
			ND &C = new_version ? D.back() : old;
			P rr = getref(C);
			if (QL <= l and r <= QR)
			{
				C.alz *= TMP;
				C.v *= TMP;
				C.mlz *= TMP;
				return rr;
			}
			pushdown(C, l, r);
			int m = mid(l, r);
			if (QL <= m)
				resolve(rr).l = _updatem(l, m, C.l);
			if (m + 1 <= QR)
				resolve(rr).r = _updatem(m + 1, r, C.r);
			maintain(C);
			return rr;
		}
		/* 区间乘法，head写时间，如果是最近一次则填H.back() */
		void updatem(int l, int r, T val, P head, bool new_ver = true)
		{
			TMP = val;
			QL = l;
			QR = r;
			new_version = new_ver;
			if (not new_ver)
				_updatem(LB, RB, head);
			else
				H.emplace_back(_updatem(LB, RB, head));
		}
		P _updatea(int l, int r, P o)
		{
			ND &old = resolve(o);
			if (new_version)
				D.emplace_back(old);
			ND &C = new_version ? D.back() : old;
			P rr = getref(C);
			if (QL <= l and r <= QR)
			{
				int len = r - l + 1;
				C.alz += TMP;
				// T tp = TMP;
				// tp *= len;
				C.v += TMP * len;
				return rr;
			}
			pushdown(C, l, r);
			int m = mid(l, r);
			if (QL <= m)
				C.l = _updatea(l, m, C.l);
			if (m + 1 <= QR)
				C.r = _updatea(m + 1, r, C.r);
			maintain(C);
			return rr;
		}
		/* 区间加法，head写时间，如果是最近一次则填H.back() */
		void updatea(int l, int r, T val, P head, bool new_ver = true)
		{
			TMP = val;
			QL = l;
			QR = r;
			new_version = new_ver;
			if (not new_ver)
				_updatea(LB, RB, head);
			else
				H.emplace_back(_updatea(LB, RB, head));
		}
		T _query(int l, int r, P p)
		{
			ND &C = resolve(p);
			if (QL <= l and r <= QR)
				return C.v;
			pushdown(C, l, r);
			T res = 0;
			int m = mid(l, r);
			if (QL <= m)
				res += _query(l, m, C.l);
			if (QR >= m + 1)
				res += _query(m + 1, r, C.r);
			return res;
		}

		T query(int l, int r, P head)
		{
			QL = l;
			QR = r;
			return _query(LB, RB, head);
		}

		/* 从0开始的区间第k大，左开右闭，填H数组的对应位置 */
		int kth(T k, P l, P r)
		{
			QL = LB;
			QR = RB;
			while (QL < QR)
			{
				ND &u = resolve(l);
				ND &v = resolve(r);
				T elem = resolve(v.l).v - resolve(u.l).v;
				int m = mid(QL, QR);
				if (elem > k)
				{
					QR = m;
					l = u.l;
					r = v.l;
				}
				else
				{
					QL = m + 1;
					k -= elem;
					l = u.r;
					r = v.r;
				}
			}
			return QL;
		}
	};
};