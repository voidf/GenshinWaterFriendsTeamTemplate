#ifndef Geo_Polygon3_H
#define Geo_Polygon3_H

#include "Geo_Base.cpp"
#include "Geo_vec3.cpp"
#include "Geo_Face3.cpp"

namespace Geometry
{
    struct Polygon3
	{
		std::vector<vec3> points;

		inline vec3 average()
		{
			vec3 avg(0);
			for (auto i : points)
				avg += i;
			return avg / points.size();
		}
		/* n^2增量法三维凸包，返回面列表(下标顶点引用) */
		inline std::vector<std::array<int, 3>> ConvexHull()
		{
			for (auto &i : points)
			{
				i.x += randreal(-decimal_round, decimal_round);
				i.y += randreal(-decimal_round, decimal_round);
				i.z += randreal(-decimal_round, decimal_round);
			}
			std::vector<std::array<int, 3>> rf, rC;
			std::vector<std::vector<char>> vis(points.size(), std::vector<char>(points.size()));
			rf.emplace_back(std::array<int, 3>({0, 1, 2}));
			rf.emplace_back(std::array<int, 3>({2, 1, 0}));
			int cnt = 2;
			for (int i = 3, cc = 0; i < points.size(); ++i)
			{
				bool vi;
				int cct = 0;
				for (auto &j : rf)
				{
					if (!(vi = Face3::visible(points[j[0]], points[j[1]], points[j[2]], points[i])))
						rC.emplace_back(rf[cct]);
					for (int k = 0; k < 3; ++k)
						vis[j[k]][j[(k + 1) % 3]] = vi;
					++cct;
				}
				for (auto &j : rf)
					for (int k = 0; k < 3; ++k)
					{
						int x = j[k];
						int y = j[(k + 1) % 3];
						if (vis[x][y] and not vis[y][x])
							rC.emplace_back(std::array<int, 3>({x, y, i}));
					}
				swap(rf, rC);
				rC.clear();
			}
			return rf;
		}
	};

}

#endif
