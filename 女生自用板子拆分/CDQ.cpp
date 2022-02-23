// CDQ做三维偏序板题P3810
// 排序做第一维，CDQ做第二维，树状数组做第三维
template <typename T>
struct BIT : vector<T> // 初始化的时候记得开大1的空间
{
	using vector<T>::vector;
	static int lowb(int x) { return x & -x; }
	void upd(int i, const T &x)
	{
		for (; i < this->size(); i += lowb(i))
			this->at(i) += x;
	}
	T query(int i)
	{
		T ans = 0;
		for (; i; i -= lowb(i))
			ans += this->at(i);
		return ans;
	}
};
struct info
{
	std::array<int, 3> v;
	int ans = 0, ctr = 1;
	bool operator<(const info &y) const { return v < y.v; }
	template <int D>
	static bool comparator(const info &x, const info &y) { return x.v[D] < y.v[D]; }
};

int unique(vector<info> &li) // 返回有效个数
{
	int cnt = 0;
	for (int i = 1; i < li.size(); ++i)
		if (li[i].v == li[cnt].v)
			++li[cnt].ctr; // 如果遇到与 i 一样的，x值就要自加一
		else
			li[++cnt] = li[i];
	return cnt + 1;
}
void solve()
{
	int n, k;
	qr(n);
	qr(k);
	BIT<int> bit(int(k) + 1);
	vector<info> li(n), tmp(n); // tmp: 按y归并排序的info数组
	for (auto &i : li)
		for (auto &j : i.v)
			qr(j);
	sort(li.begin(), li.end()); // 排x
	int cnt = unique(li);
	li.resize(cnt);
	using P = vector<info>::iterator;

	function<void(P, P)> cdq = [&](P l, P r) {
		if (r - l <= 1)
			return;
		auto mid = l + ((r - l) >> 1);
		auto p = l, q = mid, len = tmp.begin();
		cdq(l, mid);
		cdq(mid, r);
		while (p < mid and q < r)
			if (p->v[1] <= q->v[1])							 // 双指针划y
				bit.upd(p->v[2], p->ctr), *(len++) = *(p++); // 树状数组处理z
			else
				q->ans += bit.query(q->v[2]), *(len++) = *(q++);
		while (p < mid)
			bit.upd(p->v[2], p->ctr), *(len++) = *(p++);
		while (q < r)
			q->ans += bit.query(q->v[2]), *(len++) = *(q++);
		for (auto t = l; t < mid; ++t)
			bit.upd(t->v[2], -t->ctr);
		copy(tmp.begin(), len, l);
	};
	cdq(li.begin(), li.end());
	vector<int> A(n);
	for (auto &i : li)
		A[i.ans + i.ctr - 1] += i.ctr;
	for (auto &i : A)
		cout << i << endl;
};