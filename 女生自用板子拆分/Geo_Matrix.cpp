#include "Geo_Base.cpp"
#include "Geo_VectorN.cpp"

namespace Geometry
{

    template <typename VALUETYPE = FLOAT_>
    struct Matrix : VectorN<VectorN<VALUETYPE>>
    {
        int ROW, COL;

		inline std::string ToString() const
		{
			std::ostringstream ostr;
			ostr << "Matrix" << ROW << "x" << COL << "[\n";
			for (auto &i : *this)
				ostr << '\t' << i.ToString();
			ostr << "]";
			return ostr.str();
		}

        inline friend std::ostream &operator<<(std::ostream &o, const Matrix &m) { return o << m.ToString(); }

        Matrix(const VectorN<VectorN<VALUETYPE>> &v) : VectorN<VectorN<VALUETYPE>>(v), ROW(v.size()), COL(v.front().size()) {}

        Matrix(int r, int c, const VALUETYPE &default_val = 0) : ROW(r), COL(c)
        {
            this->resize(r);
            for (r--; r >= 0; r--)
                (*this)[r].init(c, default_val);
        }
        Matrix() = default;

        /* 交换两行 */
		inline void swap_rows(int from, int to) { std::swap((*this)[from], (*this)[to]); }


        /* 化为上三角矩阵 */
        inline void triangularify(bool unitriangularify = false)
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

        /* 化为行最简型 */
        void row_echelonify()
        {
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

        /* 返回一个自身化为上三角矩阵的拷贝 */
        inline Matrix triangular(bool unitriangularify = false) const
        {
            Matrix ret(*this);
            ret.triangularify(unitriangularify);
            return ret;
        }

        /* 求秩，得先上三角化 */
        inline int _rank() const
        {
            int res = 0;
            for (auto &i : (*this))
                res += i.any();
            return res;
        }

        /* 求秩 */
        int rank() const { return triangular()._rank(); }

        /* 高斯消元解方程组 */
        inline bool solve()
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

        /* 矩阵连接 */
        inline void rconcat(const Matrix &rhs)
        {
            COL += rhs.COL;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].rconcat(rhs[i]);
            }
        }

        /* 左截断 */
        inline void lerase(int ctr)
        {
            // assert(COL >= ctr);
            COL -= ctr;
            for (int i = 0; i < ROW; i++)
            {
                (*this)[i].lerase(ctr);
            }
        }

        /* 矩阵乘法 */
		inline Matrix dot(const Matrix &rhs) const
		{
			// if (this->COL != rhs.ROW)
			// throw "Error at matrix multiply: lhs's column is not equal to rhs's row";
			Matrix ret(this->ROW, rhs.COL, 0);
			for (int i = 0; i < ret.ROW; ++i)
				for (int k = 0; k < this->COL; ++k)
				{
					const VALUETYPE &s = (*this)[i][k];
					for (int j = 0; j < ret.COL; ++j)
						ret[i][j] += s * rhs[k][j];
				}
			return ret;
		}

        inline Matrix operator*(const Matrix &rhs) const { return dot(rhs); }
		inline Matrix &operator*=(const Matrix &rhs) { return (*this) = dot(rhs); }
		inline Matrix &operator*=(const VALUETYPE &rhs)
		{
			for (auto &i : *this)
				i *= rhs;
			return (*this);
		}
		inline Matrix operator*(const VALUETYPE &rhs) const { return Matrix(*this) *= rhs; }
		inline friend Matrix operator*(const VALUETYPE &rhs, const Matrix &mat) { return mat * rhs; }
		inline Matrix operator+(Matrix rhs) const
		{
			for (int i = 0; i < ROW; ++i)
				for (int j = 0; j < COL; ++j)
					rhs[i][j] += (*this)[i][j];
			return rhs;
		}
		inline Matrix &operator+=(const Matrix &rhs)
		{
			for (int i = 0; i < ROW; ++i)
				for (int j = 0; j < COL; ++j)
					(*this)[i][j] += rhs[i][j];
			return *this;
		}
    };

}