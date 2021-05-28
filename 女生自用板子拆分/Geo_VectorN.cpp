#include "Geo_Base.cpp"

namespace Geometry
{
    template <typename VALUETYPE = FLOAT_>
    struct VectorN : std::vector<VALUETYPE>
    {
        void init(int siz, VALUETYPE default_val = 0)
        {
            this->clear();
            this->assign(siz, default_val);
            this->resize(siz);
        }
        VectorN(int siz, VALUETYPE default_val = 0) { init(siz, default_val); }
        VectorN() = default;

        bool any()
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
        bool is_all_nan()
        {
            for (auto &i : *this)
            {
                if (!isnan(i))
                    return false;
            }
            return true;
        }

        void rconcat(VectorN &&r)
        {
            this->insert(this->end(), r.begin(), r.end());
        }
        void rconcat(VectorN &r) { rconcat(std::move(r)); }

        void lerase(int ctr)
        {
            assert(this->size() >= ctr);
            this->erase(this->begin(), this->begin() + ctr);
        }

        // 四则运算赋值重载
        VectorN &operator*=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i *= operand;
            return (*this);
        }
        VectorN &operator*=(VALUETYPE &operand) { return (*this) *= std::move(operand); }

        VectorN &operator/=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i /= operand;
            return (*this);
        }
        VectorN &operator/=(VALUETYPE &operand) { return (*this) /= std::move(operand); }

        VectorN &operator%=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i = (i % operand + operand) % operand;
            return (*this);
        }
        VectorN &operator%=(VALUETYPE &operand) { return (*this) %= std::move(operand); }

        VectorN &operator-=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i -= operand;
            return (*this);
        }
        VectorN &operator-=(VALUETYPE &operand) { return (*this) -= std::move(operand); }

        VectorN &operator+=(VALUETYPE &&operand)
        {
            for (VALUETYPE &i : *this)
                i += operand;
            return (*this);
        }
        VectorN &operator+=(VALUETYPE &operand) { return (*this) += std::move(operand); }
        // 结束

        // 四则运算 和单个数

        VectorN operator*(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * operand;
            return ret;
        }
        VectorN operator*(VALUETYPE &operand) { return (*this) * std::move(operand); }
        friend VectorN operator*(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand * r[i];
            return ret;
        }
        friend VectorN operator*(VALUETYPE &operand, VectorN &r) { return std::move(operand) * r; }

        VectorN operator/(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / operand;
            return ret;
        }
        VectorN operator/(VALUETYPE &operand) { return (*this) / std::move(operand); }
        friend VectorN operator/(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand / r[i];
            return ret;
        }
        friend VectorN operator/(VALUETYPE &operand, VectorN &r) { return std::move(operand) / r; }

        VectorN operator+(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + operand;
            return ret;
        }
        VectorN operator+(VALUETYPE &operand) { return (*this) + std::move(operand); }
        friend VectorN operator+(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand + r[i];
            return ret;
        }
        friend VectorN operator+(VALUETYPE &operand, VectorN &r) { return std::move(operand) + r; }

        VectorN operator-(VALUETYPE &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - operand;
            return ret;
        }
        VectorN operator-(VALUETYPE &operand) { return (*this) - std::move(operand); }
        friend VectorN operator-(VALUETYPE &&operand, VectorN &r)
        {
            VectorN ret(r.size());
            for (int i = 0; i < r.size(); i++)
                ret[i] = operand - r[i];
            return ret;
        }
        friend VectorN operator-(VALUETYPE &operand, VectorN &r) { return std::move(operand) - r; }

        /*不推荐使用的转发算子*/
        template <typename ANYRHS>
        VectorN operator+(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator-(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator*(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * rhs;
            return ret;
        }
        template <typename ANYRHS>
        VectorN operator/(ANYRHS rhs)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / rhs;
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator+(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs + rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator-(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs - rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator*(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs * rhs[i];
            return ret;
        }
        template <typename ANYRHS>
        friend VectorN operator/(ANYRHS lhs, VectorN &rhs)
        {
            VectorN ret(rhs.size());
            for (int i = 0; i < rhs.size(); i++)
                ret[i] = lhs / rhs[i];
            return ret;
        }

        // 结束

        // 四则运算 和同类

        VectorN operator+(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] + operand[i];
            return ret;
        }
        VectorN operator+(VectorN &operand) { return *this + std::move(operand); }

        VectorN operator-(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] - operand[i];
            return ret;
        }
        VectorN operator-(VectorN &operand) { return *this - std::move(operand); }

        VectorN operator*(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] * operand[i];
            return ret;
        }
        VectorN operator*(VectorN &operand) { return *this * std::move(operand); }

        VectorN operator/(VectorN &&operand)
        {
            VectorN ret(this->size());
            for (int i = 0; i < this->size(); i++)
                ret[i] = (*this)[i] / operand[i];
            return ret;
        }
        VectorN operator/(VectorN &operand) { return *this / std::move(operand); }

        // 结束

        // 赋值算子

        VectorN &operator+=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] += operand[i];
            return (*this);
        }
        VectorN &operator-=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] -= operand[i];
            return (*this);
        }
        VectorN &operator*=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] *= operand[i];
            return (*this);
        }
        VectorN &operator/=(VectorN &&operand)
        {
            for (int i = 0; i < this->size(); i++)
                (*this)[i] /= operand[i];
            return (*this);
        }

        VectorN &operator+=(VectorN &operand) { return (*this) += std::move(operand); }
        VectorN &operator-=(VectorN &operand) { return (*this) -= std::move(operand); }
        VectorN &operator*=(VectorN &operand) { return (*this) *= std::move(operand); }
        VectorN &operator/=(VectorN &operand) { return (*this) /= std::move(operand); }

        std::string ToString()
        {
            std::ostringstream ostr;
            for (int i = 0; i < this->size(); i++)
            {
                ostr << std::setw(8) << std::scientific << std::right;
                ostr << (*this)[i] << (i == this->size() - 1 ? "\n" : ", ");
            }
            return ostr.str();
        }
        friend std::ostream &operator<<(std::ostream &o, VectorN &m) { return o << m.ToString(); }
        friend std::ostream &operator<<(std::ostream &o, VectorN &&m) { return o << m.ToString(); }

        /* 向量的平方模 */
        VALUETYPE sqrMagnitude()
        {
            VALUETYPE res = 0;
            for (auto &i : (*this))
                res += i * i;
            return res;
        }

        /* 向量的模 */
        VALUETYPE magnitude() { return sqrt(this->sqrMagnitude()); }

        /* 向量单位化 */
        void Normalize()
        {
            VALUETYPE _m = this->magnitude();
            for (auto &i : (*this))
                i /= _m;
        }

        /* 返回与该向量方向同向的单位向量 */
        VectorN normalized()
        {
            VectorN ret(*this);
            ret.Normalize();
            return ret;
        }

        /* 距离 */
        static VALUETYPE Distance(VectorN &a, VectorN &b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &a, VectorN &&b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &&a, VectorN &b) { return (a - b).magnitude(); }
        static VALUETYPE Distance(VectorN &&a, VectorN &&b) { return (a - b).magnitude(); }

        /*向量线性插值*/
        static VectorN LerpUnclamped(VectorN &a, VectorN &b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &a, VectorN &&b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &&a, VectorN &b, VALUETYPE t) { return a + (b - a) * t; }
        static VectorN LerpUnclamped(VectorN &&a, VectorN &&b, VALUETYPE t) { return a + (b - a) * t; }

        /* 点积 */
        static VALUETYPE Dot(VectorN &lhs, VectorN &rhs) { return Dot(std::move(lhs), std::move(rhs)); }
        static VALUETYPE Dot(VectorN &lhs, VectorN &&rhs) { return Dot(std::move(lhs), rhs); }
        static VALUETYPE Dot(VectorN &&lhs, VectorN &rhs) { return Dot(lhs, std::move(rhs)); }
        static VALUETYPE Dot(VectorN &&lhs, VectorN &&rhs)
        {
            VALUETYPE ans = 0;
            for (auto i = 0; i < lhs._len; i++)
                ans += lhs.data[i] * rhs.data[i];
            return ans;
        }

        /*无符号弧度夹角*/
        static VALUETYPE Rad(VectorN &from, VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &from, VectorN &&to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &&from, VectorN &to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }
        static VALUETYPE Rad(VectorN &&from, VectorN &&to) { return acos(VectorN::Dot(from, to) / (from.magnitude() * to.magnitude())); }

        /*无符号角度夹角*/
        static VALUETYPE Angle(VectorN &from, VectorN &to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &from, VectorN &&to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &&from, VectorN &to) { return Rad(from, to) * 180.0 / PI; }
        static VALUETYPE Angle(VectorN &&from, VectorN &&to) { return Rad(from, to) * 180.0 / PI; }

        /*返回俩向量中x的最大值和y的最大值构造而成的向量*/
        static VectorN Max(VectorN lhs, VectorN &&rhs)
        {
            for (auto &&i : range(lhs._len))
                lhs.data[i] = std::max(lhs.data[i], rhs.data[i]);
            return lhs;
        }
        static VectorN Max(VectorN lhs, VectorN &rhs) { return Max(lhs, std::move(rhs)); }

        /*返回俩向量中x的最小值和y的最小值构造而成的向量*/
        static VectorN Min(VectorN lhs, VectorN &&rhs)
        {
            for (auto &&i : range(lhs._len))
                lhs.data[i] = std::min(lhs.data[i], rhs.data[i]);
            return lhs;
        }
        static VectorN Min(VectorN lhs, VectorN &rhs) { return Min(lhs, std::move(rhs)); }

        /*获得vector在onNormal方向的投影*/
        static VectorN Project(VectorN &vector, VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &vector, VectorN &&onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &&vector, VectorN &onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
        static VectorN Project(VectorN &&vector, VectorN &&onNormal) { return cos(Rad(vector, onNormal)) * vector.magnitude() * onNormal; }
    };

}