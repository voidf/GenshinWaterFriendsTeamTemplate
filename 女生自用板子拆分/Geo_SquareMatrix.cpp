#include "Geo_Base.cpp"
#include "Geo_VectorN.cpp"
#include "Geo_Matrix.cpp"

namespace Geometry
{
    
    template <typename VALUETYPE = FLOAT_>
    struct SquareMatrix : Matrix<VALUETYPE>
    {
        SquareMatrix(int siz, VALUETYPE &&default_val = 0) : Matrix<VALUETYPE>(siz, siz, default_val) {}
        SquareMatrix(Matrix<VALUETYPE> &&x) : Matrix<VALUETYPE>(x) { assert(x.COL == x.ROW); }
        SquareMatrix(Matrix<VALUETYPE> &x) : Matrix<VALUETYPE>(x) { assert(x.COL == x.ROW); }
        static SquareMatrix eye(int siz)
        {
            SquareMatrix ret(siz);
            for (siz--; siz >= 0; siz--)
                ret[siz][siz] = 1;
            return ret;
        }

        SquareMatrix quick_power(long long p, long long mod = 0)
        {
            SquareMatrix ans = eye(this->ROW);
            SquareMatrix rhs(*this);
            while (p)
            {
                if (p & 1)
                {
                    ans = ans.dot(rhs, mod);
                }
                rhs = rhs.dot(rhs, mod);
                p >>= 1;
            }
            return ans;
        }

        SquareMatrix inv(long long mod = 0)
        {
            Matrix<VALUETYPE> ret(*this);
            ret.rconcat(eye(this->ROW));
            ret.row_echelonify(mod);
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