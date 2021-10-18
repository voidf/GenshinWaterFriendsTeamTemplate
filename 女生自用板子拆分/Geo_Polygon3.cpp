#include "Geo_Base.cpp"
#include "Geo_Vector3.cpp"

namespace Geometry
{
    struct Polygon3
    {
        std::vector<Vector3> points;

        Vector3 average()
        {
            Vector3 avg(0, 0);
            for (auto i : points)
                avg += i;
            return avg / points.size();
        }
    };

}