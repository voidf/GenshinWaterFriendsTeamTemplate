#include "Headers.cpp"

template <size_t R, size_t C, typename T = int>
struct StaticMatrix : std::array<std::array<T, C>, R>
{
	std::string ToString() const
	{
		std::ostringstream ostr;
		ostr << "StaticMatrix" << R << "x" << C << "[\n";
		for (auto &i : *this)
		{
			for (auto &j : i)
				ostr << '\t' << j;
			ostr << "\n";
		}
		ostr << "]";
		return ostr.str();
	}

	friend std::ostream &operator<<(std::ostream &o, StaticMatrix &m) { return o << m.ToString(); }
	friend std::ostream &operator<<(std::ostream &o, StaticMatrix &&m) { return o << m.ToString(); }

	inline static StaticMatrix eye()
	{
		static_assert(R == C);
		StaticMatrix ret;
		for (int i = 0; i < R; ++i)
			ret[i][i] = 1;
		return ret;
	}
	/* 交换两行 */
	inline void swap_rows(const int from, const int to) { std::swap((*this)[from], (*this)[to]); }

	/* 化为上三角矩阵 */
	inline void triangularify(bool unitriangularify = false)
	{
		int mx;
		int done_rows = 0;
		for (int j = 0; j < C; j++) // 化为上三角
		{
			mx = done_rows;
			for (int i = done_rows + 1; i < R; i++)
			{
				if (fabs((*this)[i][j]) > fabs((*this)[mx][j]))
					mx = i;
			}
			if ((*this)[mx][j] == 0)
				continue;
			if (mx != done_rows)
				swap_rows(mx, done_rows);

			for (int i = done_rows + 1; i < R; i++)
			{
				T tmp = (*this)[i][j] / (*this)[done_rows][j];
				if (tmp != 0)
					for (int k = 0; k < C; ++k)
						(*this)[i][k] -= (*this)[done_rows][k] * tmp;
			}
			if (unitriangularify)
			{
				auto tmp = (*this)[done_rows][j];
				for (int k = 0; k < C; ++k)
					(*this)[done_rows][k] /= tmp; // 因为用了引用，这里得拷贝暂存
			}
			done_rows++;
			if (done_rows == R)
				break;
		}
	}

	/* 化为行最简型 */
	inline void row_echelonify()
	{
		triangularify(true);
		int valid_pos = 1;
		for (int i = 1; i < R; i++)
		{
			while (valid_pos < C and (*this)[i][valid_pos] == 0)
				valid_pos++;
			if (valid_pos == C)
				break;
			for (int ii = i - 1; ii >= 0; ii--)
			{
				for (int jj = 0; jj < C; ++jj)
					(*this)[ii][jj] -= (*this)[i][jj] * (*this)[ii][valid_pos];
			}
		}
	}

	/* 返回一个自身化为上三角矩阵的拷贝 */
	inline StaticMatrix triangular(bool unitriangularify = false) const
	{
		StaticMatrix ret(*this);
		ret.triangularify(unitriangularify);
		return ret;
	}

	/* 求秩，得先上三角化 */
	inline int _rank() const
	{
		int res = 0;
		for (auto &i : (*this))
			res += (i.back() != 0);
		return res;
	}

	/* 求秩 */
	inline int rank() const { return triangular()._rank(); }

	/* 高斯消元解方程组 */
	inline bool solve()
	{
		triangularify();
		if (!(*this).back().back())
			return false;
		for (int i = R - 1; i >= 0; i--)
		{
			for (int j = i + 1; j < R; j++)
				(*this)[i][C - 1] -= (*this)[i][j] * (*this)[j][C - 1];
			if ((*this)[i][i] == 0)
				return false;
			(*this)[i][C - 1] /= (*this)[i][i];
		}
		return true;
	}

	/* 矩阵乘法 */
	template <size_t _C>
	inline StaticMatrix<R, _C, T> dot(const StaticMatrix<C, _C, T> &rhs) const
	{
		StaticMatrix<R, _C, T> ret;
		for (int i = 0; i < R; ++i)
			for (int k = 0; k < C; ++k)
			{
				const T &s = (*this)[i][k];
				for (int j = 0; j < _C; ++j)
					ret[i][j] += s * rhs[k][j];
			}
		return ret;
	}
	inline bool operator!=(const StaticMatrix &rhs) const
	{
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < C; ++j)
				if ((*this)[i][j] != rhs[i][j])
					return true;
		return false;
	}
	inline bool operator==(const StaticMatrix &rhs) const { return !(*this == rhs); }
	template <size_t _C>
	inline StaticMatrix<R, _C, T> operator*(const StaticMatrix<C, _C, T> &rhs) const { return dot(rhs); }
	template <size_t _C>
	inline StaticMatrix<R, _C, T> &operator*=(const StaticMatrix<C, _C, T> &rhs) { return (*this) = dot(rhs); }
	inline StaticMatrix &operator+=(const StaticMatrix &rhs)
	{
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < C; ++j)
				(*this)[i][j] += rhs[i][j];
		return *this;
	}
	inline StaticMatrix &operator+=(const T &rhs)
	{
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < C; ++j)
				(*this)[i][j] += rhs;
		return *this;
	}
	inline StaticMatrix operator+(const StaticMatrix &rhs) const { return StaticMatrix(*this) += rhs; }
	inline friend StaticMatrix operator+(const T &rhs, StaticMatrix mat) { return mat + rhs; }
	inline StaticMatrix &operator*=(const T &rhs)
	{
		for (auto &i : (*this))
			for (auto &j : i)
				j *= rhs;
		return (*this);
	}
	inline StaticMatrix operator*(const T &rhs) const { return StaticMatrix(*this) *= rhs; }
	inline friend StaticMatrix operator*(const T &rhs, StaticMatrix mat) { return mat * rhs; }
};