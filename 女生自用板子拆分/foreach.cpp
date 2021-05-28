#include "Headers.cpp"

struct range
{
    struct _iterator
    {
        LL _start;
        LL _step;
        _iterator(LL from, LL step) : _start(from), _step(step) {}
        _iterator(LL from) : _start(from), _step(1) {}
        inline bool sign(LL x) { return x < 0; }
        bool operator!=(_iterator &b) { return _start != b._start and sign(b._start - _start) == sign(_step); }
        LL operator*() { return _start; }
        _iterator &operator++()
        {
            _start += _step;
            return *this;
        }
    };
    _iterator _finish;
    _iterator _begin;
    range(LL to) : _begin(0), _finish(to) {}
    range(LL from, LL to) : _begin(from), _finish(to) {}
    range(LL from, LL to, LL step) : _begin(from, step), _finish(to, step) {}
    _iterator &begin() { return _begin; }
    _iterator &end() { return _finish; }
};

struct subset
{
    struct _iterator
    {
        LL _start;
        LL _father;
        _iterator(LL from, LL step) : _start(from), _father(step) {}
        bool operator!=(_iterator &b) { return _start != b._start; }
        LL operator*() { return _start; }
        _iterator &operator++()
        {
            _start = (_start - 1) & _father;
            return *this;
        }
    };
    _iterator _finish;
    _iterator _begin;
    subset(LL father) : _begin(father, father), _finish(0, father) {}
    _iterator &begin() { return _begin; }
    _iterator &end() { return _finish; }
};