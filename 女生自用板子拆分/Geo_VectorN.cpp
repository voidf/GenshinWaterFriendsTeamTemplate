#include "Geo_Base.cpp"

namespace Geometry
{
    template <typename VALUETYPE = FLOAT_>
    struct VectorN : std::vector<VALUETYPE>
    {
        inline void init(int siz, const VALUETYPE &default_val = 0)
        {
            this->assign(siz, default_val);
            this->resize(siz);
        }
        VectorN() = default;

        inline bool any() const
        {
            for (auto &i : *this)
            {
                if (i != 0)
                    return true;
                else if (!isfinite(i))
                    return false;
            }
            return false;
        }
        inline bool is_all_nan() const
        {
            for (auto &i : *this)
            {
                if (!isnan(i))
                    return false;
            }
            return true;
        }

        inline void rconcat(const VectorN &r) { this->insert(this->end(), r.begin(), r.end()); }

        inline void lerase(int ctr) { this->erase(this->begin(), this->begin() + ctr); }

        // 四则运算赋值重载
        inline VectorN &operator*=(const VALUETYPE &operand)
        {
            for (VALUETYPE &i : *this)
                i *= operand;
            return (*this);
        }

        inline VectorN &operator/=(const VALUETYPE &operand)
        {
            for (VALUETYPE &i : *this)
                i /= operand;
            return (*this);
        }

        inline VectorN &operator%=(const VALUETYPE &operand)
        {
            for (VALUETYPE &i : *this)
                i = i % operand;
            return (*this);
        }

        inline VectorN &operator-=(const VALUETYPE &operand)
        {
            for (VALUETYPE &i : *this)
                i -= operand;
            return (*this);
        }

        inline VectorN &operator+=(const VALUETYPE &operand)
        {
            for (VALUETYPE &i : *this)
                i += operand;
            return (*this);
        }
        // 结束

        // 四则运算 和单个数
        inline VectorN operator*(const VALUETYPE &operand) const { return VectorN(*this) *= operand; }
        inline friend VectorN operator*(const VALUETYPE &operand, const VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand * r[i];
            return ret;
        }

        inline VectorN operator/(const VALUETYPE &operand) const { return VectorN(*this) /= operand; }
        inline friend VectorN operator/(const VALUETYPE &operand, const VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand / r[i];
            return ret;
        }

        inline VectorN operator+(const VALUETYPE &operand) const { return VectorN(*this) += operand; }
        inline friend VectorN operator+(const VALUETYPE &operand, const VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand + r[i];
            return ret;
        }

        inline VectorN operator-(const VALUETYPE &operand) const { return VectorN(*this) -= operand; }
        inline friend VectorN operator-(const VALUETYPE &operand, const VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand - r[i];
            return ret;
        }
        /*不推荐使用的转发算子*/
        template <typename ANYRHS>
        inline VectorN operator+(const ANYRHS &rhs) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + rhs;
            return ret;
        }
        template <typename ANYRHS>
        inline VectorN operator-(const ANYRHS &rhs) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - rhs;
            return ret;
        }
        template <typename ANYRHS>
        inline VectorN operator*(const ANYRHS &rhs) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * rhs;
            return ret;
        }
        template <typename ANYRHS>
        inline VectorN operator/(const ANYRHS &rhs) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / rhs;
            return ret;
        }
        template <typename ANYRHS>
        inline friend VectorN operator+(const ANYRHS &lhs, const VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs + rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        inline friend VectorN operator-(const ANYRHS &lhs, const VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs - rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        inline friend VectorN operator*(const ANYRHS &lhs, const VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs * rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        inline friend VectorN operator/(const ANYRHS &lhs, const VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs / rhs[i];
            return ret;
        }

        // 结束

        // 四则运算 和同类

        inline VectorN operator+(const VectorN &operand) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + operand[i];
            return ret;
        }

        inline VectorN operator-(const VectorN &operand) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - operand[i];
            return ret;
        }

        inline VectorN operator*(const VectorN &operand) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * operand[i];
            return ret;
        }

        inline VectorN operator/(const VectorN &operand) const
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / operand[i];
            return ret;
        }
        // 结束

        // 赋值算子

        inline VectorN &operator+=(const VectorN &operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] += operand[i];
            return (*this);
        }
        inline VectorN &operator-=(const VectorN &operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] -= operand[i];
            return (*this);
        }
        inline VectorN &operator*=(const VectorN &operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] *= operand[i];
            return (*this);
        }
        inline VectorN &operator/=(const VectorN &operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] /= operand[i];
            return (*this);
        }

        inline std::string ToString() const
        {
            std::ostringstream ostr;
            for (int i = 0; i < this->size(); i++)
            {
                ostr << std::setw(8) << std::scientific << std::right;
                ostr << (*this)[i] << (i == this->size() - 1 ? "\n" : ", ");
            }
            return ostr.str();
        }
        inline friend std::ostream &operator<<(std::ostream &o, const VectorN &m) { return o << m.ToString(); }

        /* 向量的平方模 */
        inline VALUETYPE sqrMagnitude() const
        {
            VALUETYPE res = 0;
            for (auto &i : (*this))
                res += i * i;
            return res;
        }

        /* 向量的模 */
        inline VALUETYPE magnitude() const { return sqrt(this->sqrMagnitude()); }

        /* 向量单位化 */
        inline void Normalize()
        {
            VALUETYPE _m = this->magnitude();
            for (auto &i : (*this))
                i /= _m;
        }

        /* 返回与该向量方向同向的单位向量 */
        inline VectorN normalized() const
        {
            VectorN ret(*this);
            ret.Normalize();
            return ret;
        }

        /* 距离 */
        inline static VALUETYPE Distance(const VectorN &a, const VectorN &b) { return (a - b).magnitude(); }

        /* 向量线性插值 */
        inline static VectorN LerpUnclamped(const VectorN &a, const VectorN &b, const VALUETYPE &t) { return a + (b - a) * t; }

        /* 点积 */
        inline static VALUETYPE Dot(const VectorN &lhs, const VectorN &rhs)
        {
            VALUETYPE ans = 0;
            for (auto i = 0; i < lhs.size(); i++)
                ans += lhs[i] * rhs[i];
            return ans;
        }

        /* 无符号弧度夹角 */
        inline static VALUETYPE Rad(const VectorN &from, const VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /* 无符号角度夹角 */
        inline static VALUETYPE Angle(const VectorN &from, const VectorN &to) { return Rad(from, to) * 180.0 / PI; }

        /* 返回俩向量中x的最大值和y的最大值构造而成的向量 */
        inline static VectorN Max(VectorN lhs, const VectorN &rhs)
        {
            for (int i = 0; i < lhs.size(); ++i)
                lhs[i] = std::max(lhs[i], rhs[i]);
            return lhs;
        }

        /* 返回俩向量中x的最小值和y的最小值构造而成的向量 */
        inline static VectorN Min(VectorN lhs, const VectorN &rhs)
        {
            for (int i = 0; i < lhs.size(); ++i)
                lhs[i] = std::min(lhs[i], rhs[i]);
            return lhs;
        }

        /* 获得vector在onNormal方向的投影 */
        inline static VectorN Project(const VectorN &vector, const VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
    };

}