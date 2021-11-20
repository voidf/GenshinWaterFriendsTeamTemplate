#ifndef Geo_Face3_H
#define Geo_Face3_H

#include "Geo_Base.cpp"

namespace Geometry
{

	template <typename T, int K = 2>
	struct KDTree
	{
#define NIL -1
		std::vector<std::array<T, K>> &E;
		using P = int;

		using itr = typename std::vector<std::array<T, K>>::iterator;
		// template<typename T, int K=2>
		struct Node
		{
			// using itr = std::vector<std::array<T, K>>::iterator;
			itr elem;
			int dim;
			P ls = NIL, rs = NIL;
			Node(itr _i) : elem(_i) {}
			Node(itr _i, int dimension) : elem(_i), dim(dimension) {}
		};
		std::vector<Node> D;

		P build(const itr l, const itr r, int k)
		{
			if (r - l <= 0)
				return NIL;
			if (k == K)
				k = 0;
			if (r - l == 1)
			{
				D.emplace_back(l);
				return D.size() - 1;
			}
			itr m = l + (r - l - 1) / 2;
			nth_element(l, m, r, [&](const std::array<T, K> &a, const std::array<T, K> &b) -> bool
						{ return a[k] < b[k]; });
			D.emplace_back(m);
			P nid = D.size() - 1;
			D[nid].ls = build(l, m, k + 1);
			D[nid].rs = build(m + 1, r, k + 1);
			return nid;
		}

		KDTree(int siz, std::vector<std::array<T, K>> &_E) : E(_E) { D.reserve(siz); }

		FLOAT_ gans;
		Segment2 gquery = Segment2(Vector2(0.0), Vector2(0));
		FLOAT_ mn[K], mx[K];
		void _query(P cur, int k)
		{
			if (k == K)
				k = 0;
			if (cur == NIL)
				return;
			auto &CN = D[cur];
			std::array<T, K> &karr = *CN.elem;
			Vector2 cv(karr[0], karr[1]);
			if (cv != gquery.from && cv != gquery.to)
				gans = min(gans, gquery.distToPointS(cv));
			if (karr[k] < mn[k])
			{
				if (gans > mn[k] - karr[k])
					_query(CN.ls, k + 1);
				_query(CN.rs, k + 1);
			}
			else if (karr[k] > mx[k])
			{
				if (gans > karr[k] - mx[k])
					_query(CN.rs, k + 1);
				_query(CN.ls, k + 1);
			}
			else
			{
				_query(CN.ls, k + 1);
				_query(CN.rs, k + 1);
			}
		}

		FLOAT_ query(const Segment2 &s)
		{
			// gans = INFINITY;
			gquery = s;
			for (int i = 0; i < K; ++i)
			{
				mn[i] = min(s.from.at(i), s.to.at(i));
				mx[i] = max(s.from.at(i), s.to.at(i));
				// mn[1] = min(s.from.y, s.to.y);
				// mx[1] = max(s.from.y, s.to.y);
			}
			_query(0, 0);
			return gans;
		}
	};

}

#endif