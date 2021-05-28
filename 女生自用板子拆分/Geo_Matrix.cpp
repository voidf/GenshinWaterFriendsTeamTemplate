#include "Geo_Base.cpp"
#include "Geo_VectorN.cpp"

namespace Geometry
{
    
    template <typename VALUETYPE = FLOAT_>
    struct Matrix : VectorN<VectorN<VALUETYPE>>
    {
        int ROW, COL;
        // std::vector<VectorN<VALUETYPE>> data;

        std::string ToString()
        {
            std::ostringstream ostr;
            ostr << "Matrix" << ROW << "x" << COL << "[\n";
            for (auto &i : *this)
                ostr << '\t' << i.ToString();
            ostr << "]";
            return ostr.str();
        }

        friend std::ostream &operator<<(std::ostream &o, Matrix &m) { return o << m.ToString(); }
        friend std::ostream &operator<<(std::ostream &o, Matrix &&m) { return o << m.ToString(); }

        Matrix(VectorN<VectorN<VALUETYPE>> &&v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}
        Matrix(VectorN<VectorN<VALUETYPE>> &v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}

        Matrix(int r, int c, VALUETYPE default_val = 0) : ROW(r), COL(c)
        {
            this->resize(r);
            for (r--; r >= 0; r--)
                (*this)[r].init(c, default_val);
        }
        Matrix() = default;

        /*交换两行*/
        void swap_rows(int from, int to)
        {
            std::swap((*this)[from], (*this)[to]);
        }

        /*化为上三角矩阵*/
        void triangularify(bool unitriangularify = false)
        {
            int mx;
            int done_rows = 0;
            for (int j = 0; j < COL; j++) // 化为上三角
            {
                mx = done_rows;
                for (int i = done_rows + 1; i < ROW; i++)
                {
                    if (fabs((*this)[i][j]) > fabs((*this)[mx][j]))
                        mx = i;
                }
                if ((*this)[mx][j] == 0)
                    continue;
                if (mx != done_rows)
                    swap_rows(mx, done_rows);

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    VALUETYPE tmp = (*this)[i][j] / (*this)[done_rows][j];
                    if (tmp != 0)
                        (*this)[i] -= (*this)[done_rows] * tmp;
                }
                if (unitriangularify)
                {
                    auto tmp = (*this)[done_rows][j];
                    (*this)[done_rows] /= tmp; // 因为用了引用，这里得拷贝暂存
                }
                done_rows++;
                if (done_rows == ROW)
                    break;
            }
        }

        /*化为上三角矩阵，模意义版*/
        void triangularify(long long mod, bool unitriangularify = false)
        {
            int mx;
            int done_rows = 0;
            for (int j = 0; j < COL; j++) // 化为上三角
            {
                mx = done_rows;

                if ((*this)[done_rows][j] < 0)
                    (*this)[done_rows][j] = ((*this)[done_rows][j] % mod + mod) % mod;

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    if ((*this)[i][j] < 0)
                        (*this)[i][j] = ((*this)[i][j] % mod + mod) % mod;
                    if ((*this)[i][j] > (*this)[mx][j])
                        mx = i;
                }
                if ((*this)[mx][j] == 0)
                    continue;

                if (mx != done_rows)
                    swap_rows(mx, done_rows);

                for (int i = done_rows + 1; i < ROW; i++)
                {
                    VALUETYPE tmp = (*this)[i][j] * inv((*this)[done_rows][j], mod) % mod;
                    if (tmp != 0)
                    {
                        (*this)[i] -= (*this)[done_rows] * tmp;
                        (*this)[i] %= mod;
                    }
                }
                if (unitriangularify)
                {
                    auto tmp = (*this)[done_rows][j];
                    (*this)[done_rows] *= inv(tmp, mod);
                    (*this)[done_rows] %= mod;
                }
                done_rows++;
                if (done_rows == ROW)
                    break;
            }
        }

        void row_echelonify(long long mod = 0)
        {
            if (mod)
                triangularify(mod, true);
            else
                triangularify(true);
            int valid_pos = 1;
            for (int i = 1; i < ROW; i++)
            {
                while (valid_pos < COL and (*this)[i][valid_pos] == 0)
                    valid_pos++;
                if (valid_pos == COL)
                    break;
                for (int ii = i - 1; ii >= 0; ii--)
                {
                    (*this)[ii] -= (*this)[i] * (*this)[ii][valid_pos];
                    if (mod)
                        (*this)[ii] %= mod;
                }
            }
        }

        /*返回一个自身化为上三角矩阵的拷贝*/
        Matrix triangular(bool unitriangularify = false)
        {
            Matrix ret(*this);
            ret.triangularify(unitriangularify);
            return ret;
        }

        /*求秩，得先上三角化*/
        int _rank()
        {
            int res = 0;
            for (auto &i : (*this))
                res += i.any();
            return res;
        }

        /*求秩*/
        int rank() { return triangular()._rank(); }

        /*高斯消元解方程组*/
        bool solve()
        {
            if (COL != ROW + 1)
                throw "dimension error!";
            triangularify();
            // cerr << *this << endl;
            if (!(*this).back().any())
                return false;
            for (int i = ROW - 1; i >= 0; i--)
            {
                for (int j = i + 1; j < ROW; j++)
                    (*this)[i][COL - 1] -= (*this)[i][j] * (*this)[j][COL - 1];
                if ((*this)[i][i] == 0)
                    return false;
                (*this)[i][COL - 1] /= (*this)[i][i];
            }
            return true;
        }

        /*矩阵连接*/
        void rconcat(Matrix &&rhs)
        {
            COL += rhs.COL;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].rconcat(rhs[i]);
            }
        }
        void rconcat(Matrix &rhs) { rconcat(std::move(rhs)); }

        /*左截断*/
        void lerase(int ctr)
        {
            assert(COL >= ctr);
            COL -= ctr;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].lerase(ctr);
            }
        }

        /*矩阵乘法*/
        Matrix dot(Matrix &&rhs, long long mod = 0)
        {
            if (this->COL != rhs.ROW)
                throw "Error at matrix multiply: lhs's column is not equal to rhs's row";
            Matrix ret(this->ROW, rhs.COL, true);
            for (int i = 0; i < ret.ROW; ++i)
                for (int k = 0; k < this->COL; ++k)
                {
                    VALUETYPE &s = (*this)[i][k];
                    for (int j = 0; j < ret.COL; ++j)
                    {
                        ret[i][j] += s * rhs[k][j];
                        if (mod)
                            ret[i][j] %= mod;
                    }
                }
            return ret;
        }
        Matrix dot(Matrix &rhs, long long mod = 0) { return dot(std::move(rhs), mod); }
    };

}