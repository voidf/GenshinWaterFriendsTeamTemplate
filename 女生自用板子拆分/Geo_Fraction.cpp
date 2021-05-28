#include "Geo_Base.cpp"

namespace Geometry
{
    template <typename PrecisionType = LL>
    struct Fraction
    {
        PrecisionType upper, lower;

        Fraction(PrecisionType u = 0, PrecisionType l = 1)
        {
            upper = u;
            lower = l;
        }
        void normalize()
        {
            if (upper)
            {
                PrecisionType g = abs(std::__gcd(upper, lower));
                upper /= g;
                lower /= g;
            }
            else
                lower = 1;
            if (lower < 0)
            {
                lower = -lower;
                upper = -upper;
            }
        }
        long double ToFloat() { return (long double)upper / (long double)lower; }
        bool operator==(Fraction b) { return upper * b.lower == lower * b.upper; }
        bool operator>(Fraction b) { return upper * b.lower > lower * b.upper; }
        bool operator<(Fraction b) { return upper * b.lower < lower * b.upper; }
        bool operator<=(Fraction b) { return !(*this > b); }
        bool operator>=(Fraction b) { return !(*this < b); }
        bool operator!=(Fraction b) { return !(*this == b); }
        Fraction operator-() { return Fraction(-upper, lower); }
        Fraction operator+(Fraction b) { return Fraction(upper * b.lower + b.upper * lower, lower * b.lower); }
        Fraction operator-(Fraction b) { return (*this) + (-b); }
        Fraction operator*(Fraction b) { return Fraction(upper * b.upper, lower * b.lower); }
        Fraction operator/(Fraction b) { return Fraction(upper * b.lower, lower * b.upper); }
        Fraction &operator+=(Fraction b)
        {
            *this = *this + b;
            this->normalize();
            return *this;
        }
        Fraction &operator-=(Fraction b)
        {
            *this = *this - b;
            this->normalize();
            return *this;
        }
        Fraction &operator*=(Fraction b)
        {
            *this = *this * b;
            this->normalize();
            return *this;
        }
        Fraction &operator/=(Fraction b)
        {
            *this = *this / b;
            this->normalize();
            return *this;
        }
        friend Fraction fabs(Fraction a) { return Fraction(abs(a.upper), abs(a.lower)); }
        std::string to_string() { return lower == 1 ? std::to_string(upper) : std::to_string(upper) + '/' + std::to_string(lower); }
        friend std::ostream &operator<<(std::ostream &o, Fraction a)
        {
            return o << "Fraction(" << std::to_string(a.upper) << ", " << std::to_string(a.lower) << ")";
        }
        friend std::istream &operator>>(std::istream &i, Fraction &a)
        {
            char slash;
            return i >> a.upper >> slash >> a.lower;
        }
        friend isfinite(Fraction a) { return a.lower != 0; }
        void set_value(PrecisionType u, PrecisionType d = 1) { upper = u, lower = d; }
    };

}