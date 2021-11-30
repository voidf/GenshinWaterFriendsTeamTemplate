
// #define use_ptr
// const StaticMatrix<3, 3, m1e9_7> Add0;
// const StaticMatrix<3, 3, m1e9_7> Mul1 = StaticMatrix<3, 3, m1e9_7>::eye();
#define Add0 0
#define Mul1 1
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
		T v, sqv;
		T alz;
		T mlz;
		P l = NIL;
		P r = NIL;
		Node() : v(Add0), alz(Add0), mlz(Mul1) {}
		Node(T _v) : v(_v), alz(Add0), mlz(Mul1) {}
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
			if (x.mlz != Mul1)
			{
				lson.v *= x.mlz;
				lson.alz *= x.mlz;
				lson.mlz *= x.mlz;
				rson.v *= x.mlz;
				rson.alz *= x.mlz;
				rson.mlz *= x.mlz;
				x.mlz = Mul1;
			}
			if (x.alz != Add0)
			{
				int m = mid(l, r);
				lson.v += x.alz * (m - l + 1);
				lson.alz += x.alz;
				rson.v += x.alz * (r - m);
				rson.alz += x.alz;
				x.alz = Add0;
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
			P rr = getref(D.back());
			int m = mid(l, r);
			resolve(rr).l = _build(l, m);
			resolve(rr).r = _build(m + 1, r);
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
			T res = Add0;
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

	template <class T>
	struct PST_dynamic
	{
		mutable int QL, QR;
		int LB, RB;
		using ND = Node<T>;
		std::vector<ND> D;
		std::vector<P> H;
		T *refarr;
		T MTMP;
		T ATMP;
		T RTMP;
		mutable bool new_version;
		inline ND &resolve(P x)
		{
#ifdef use_ptr
			return *x;
#else
			return D[x];
#endif
		}
		inline P getref(ND &x) const
		{
#ifdef use_ptr
			return &x;
#else
			return &x - &D.front();
#endif
		}
		PST_dynamic() {}
		PST_dynamic(int n, int m) { auto_reserve(n, m); }
		inline void auto_reserve(int n, int m)
		{
			D.reserve((1 + ceil(log2(n))) * m + 2 * n);
			H.reserve(m);
		}

		inline void maintain(ND &x)
		{
			x.v = 0;
			if (x.l != NIL)
				x.v += resolve(x.l).v;
			if (x.r != NIL)
				x.v += resolve(x.r).v;
		}

		inline void pushdown(P x, int l, int r)
		{
			if (l == r)
				return;
			int m = mid(l, r);
			if (resolve(x).l != NIL)
			{
				ND &lson = resolve(resolve(x).l);
				if (resolve(x).mlz != Mul1)
				{
					lson.v *= resolve(x).mlz;
					lson.alz *= resolve(x).mlz;
					lson.mlz *= resolve(x).mlz;
					lson.sqv *= resolve(x).mlz * resolve(x).mlz;
				}
				if (resolve(x).alz != Add0)
				{
					lson.sqv = lson.sqv + 2 * resolve(x).alz * lson.v + resolve(x).alz * resolve(x).alz * (m - l + 1);
					lson.v += resolve(x).alz * (m - l + 1);
					lson.alz += resolve(x).alz;
				}
			}
			else if (resolve(x).alz != Add0)
			{
				D.emplace_back();
				resolve(x).l = getref(D.back());
				ND &lson = resolve(resolve(x).l);
				lson.sqv = lson.sqv + 2 * resolve(x).alz * lson.v + resolve(x).alz * resolve(x).alz * (m - l + 1);
				lson.v += resolve(x).alz * (m - l + 1);
				lson.alz += resolve(x).alz;
			}
			if (resolve(x).r != NIL)
			{
				ND &rson = resolve(resolve(x).r);
				if (resolve(x).mlz != Mul1)
				{
					rson.v *= resolve(x).mlz;
					rson.alz *= resolve(x).mlz;
					rson.mlz *= resolve(x).mlz;
					rson.sqv *= resolve(x).mlz * resolve(x).mlz;
				}
				if (resolve(x).alz != Add0)
				{
					rson.sqv = rson.sqv + 2 * resolve(x).alz * rson.v + resolve(x).alz * resolve(x).alz * (r - m);
					rson.v += resolve(x).alz * (r - m);
					rson.alz += resolve(x).alz;
				}
			}
			else if (resolve(x).alz != Add0)
			{
				D.emplace_back();
				resolve(x).r = getref(D.back());
				ND &rson = resolve(resolve(x).r);
				rson.sqv = rson.sqv + 2 * resolve(x).alz * rson.v + resolve(x).alz * resolve(x).alz * (r - m);
				rson.v += resolve(x).alz * (r - m);
				rson.alz += resolve(x).alz;
			}
			resolve(x).mlz = Mul1;
			resolve(x).alz = Add0;
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
		inline void build(T *rf, int l, int r)
		{
			refarr = rf;
			LB = l;
			RB = r;
			H.emplace_back(_build(l, r));
		}

		inline void dynamic_init(int l, int r)
		{
			LB = l;
			RB = r;
			D.emplace_back();
			H.emplace_back(getref(D[0]));
		}

		P _updatem(int l, int r, P o)
		{
			if (o == NIL)
			{
				D.emplace_back();
				o = getref(D.back());
			}
			else if (new_version)
			{
				D.emplace_back(resolve(o));
				o = getref(D.back());
			}
			// ND &C = resolve(o); // 可能因为搬家出错
			if (QL <= l and r <= QR)
			{
				resolve(o).alz *= MTMP;
				resolve(o).v *= MTMP;
				resolve(o).mlz *= MTMP;
				return o;
			}
			pushdown(o, l, r);
			int m = mid(l, r);
			if (QL <= m)
			{
				auto ret = _updatem(l, m, resolve(o).l);
				resolve(o).l = ret;
			}
			if (m + 1 <= QR)
			{
				auto ret = _updatem(m + 1, r, resolve(o).r);
				resolve(o).r = ret;
			}
			maintain(resolve(o));
			return o;
		}
		/* 区间乘法，head写时间，如果是最近一次则填H.back()，new_ver不填认为当做动态开点线段树用 */
		inline void updatem(int l, int r, T val, P head = NIL, bool new_ver = true)
		{
			MTMP = val;
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
			if (o == NIL)
			{
				D.emplace_back();
				o = getref(D.back());
			}
			else if (new_version)
			{
				D.emplace_back(resolve(o));
				o = getref(D.back());
			}
			// ND &C = resolve(o);
			if (QL <= l and r <= QR)
			{
				int len = r - l + 1;
				resolve(o).sqv = resolve(o).sqv + 2 * ATMP * resolve(o).v + (ATMP * ATMP * len);
				resolve(o).alz += ATMP;
				resolve(o).v += ATMP * len;
				return o;
			}
			pushdown(o, l, r);
			int m = mid(l, r);
			if (QL <= m)
			{
				auto ret = _updatea(l, m, resolve(o).l);
				resolve(o).l = ret;
			}
			if (m + 1 <= QR)
			{
				auto ret = _updatea(m + 1, r, resolve(o).r);
				resolve(o).r = ret;
			}
			maintain(resolve(o));
			return o;
		}
		/* 区间加法，head写时间，如果是最近一次则填H.back()，new_ver不填认为当做动态开点线段树用 */
		inline void updatea(int l, int r, T val, P head = NIL, bool new_ver = true)
		{
			ATMP = val;
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
			if (p == NIL)
				return Add0;
			if (QL <= l and r <= QR)
				return resolve(p).v;
			pushdown(p, l, r);
			T res = Add0;
			int m = mid(l, r);
			if (QL <= m)
				res += _query(l, m, resolve(p).l);
			if (QR >= m + 1)
				res += _query(m + 1, r, resolve(p).r);
			return res;
		}

		inline T query(int l, int r, P head)
		{
			QL = l;
			QR = r;
			return _query(LB, RB, head);
		}

		/* 从0开始的区间第k大，左开右闭，填H数组的对应位置 */
		inline int kth(T k, P l, P r)
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

		inline int kth(T k, P head)
		{
			QL = LB;
			QR = RB;
			while (QL < QR)
			{
				ND &u = resolve(head);
				int m = mid(QL, QR);
				if (u.l == NIL)
				{
					if (u.r == NIL)
						return -1;
					head = u.r;
					QL = m + 1;
				}
				else
				{
					T &elem = resolve(u.l).v;
					if (elem > k)
					{
						QR = m;
						head = u.l;
					}
					else
					{
						if (u.r == NIL)
							return -1;
						k -= elem;
						head = u.r;
						QL = m + 1;
					}
				}
			}
			return QL;
		}

		inline int under_bound(T k, P head)
		{
			if (head == NIL)
				return -1;
			T q = query(LB, k - 1, head);
			return kth(q - 1, head);
		}

		inline int upper_bound(T k, P head)
		{
			if (head == NIL)
				return -1;
			T q = query(LB, k, head);
			return kth(q, head);
		}

		inline T rank(int x, P head) { return query(LB, x - 1, head); }
	};
};