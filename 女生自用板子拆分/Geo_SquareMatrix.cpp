#include "Geo_Base.cpp"
#include "Geo_VectorN.cpp"
#include "Geo_Matrix.cpp"

namespace Geometry
{
    
    template <typename VALUETYPE = FLOAT_>
    struct SquareMatrix : Matrix<VALUETYPE>
    {
        SquareMatrix(int siz, const VALUETYPE &default_val = 0) : Matrix<VALUETYPE>(siz, siz, default_val) {}
		SquareMatrix(const Matrix<VALUETYPE> &x) : Matrix<VALUETYPE>(x) { assert(x.COL == x.ROW); }
		inline static SquareMatrix eye(int siz)
		{
			SquareMatrix ret(siz);
			for (siz--; siz >= 0; siz--)
				ret[siz][siz] = 1;
			return ret;
		}

		inline SquareMatrix quick_power(long long p) const
		{
			SquareMatrix ans = eye(this->ROW);
			SquareMatrix rhs(*this);
			while (p)
			{
				if (p & 1)
					ans = ans.dot(rhs);
				rhs = rhs.dot(rhs);
				p >>= 1;
			}
			return ans;
		}

		inline SquareMatrix inv() const
		{
			Matrix<VALUETYPE> ret(*this);
			ret.rconcat(eye(this->ROW));
			ret.row_echelonify();
			// cerr << ret << endl;
			for (int i = 0; i < this->ROW; i++)
			{
				if (ret[i][i] != 1)
					throw "Error at matrix inverse: cannot identify extended matrix";
			}
			ret.lerase(this->ROW);
			return ret;
		}
    };

}