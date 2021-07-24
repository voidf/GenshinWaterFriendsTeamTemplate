#include "Headers.cpp"

struct rangell
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
    rangell(LL to) : _begin(0), _finish(to) {}
    rangell(LL from, LL to) : _begin(from), _finish(to) {}
    rangell(LL from, LL to, LL step) : _begin(from, step), _finish(to, step) {}
    _iterator &begin() { return _begin; }
    _iterator &end() { return _finish; }
};

struct range
{
    struct _iterator
    {
        int _start;
        int _step;
        _iterator(int from, int step) : _start(from), _step(step) {}
        _iterator(int from) : _start(from), _step(1) {}
        inline bool sign(int x) { return x < 0; }
        bool operator!=(_iterator &b) { return _start != b._start and sign(b._start - _start) == sign(_step); }
        int operator*() { return _start; }
        _iterator &operator++()
        {
            _start += _step;
            return *this;
        }
    };
    _iterator _finish;
    _iterator _begin;
    range(int to) : _begin(0), _finish(to) {}
    range(int from, int to) : _begin(from), _finish(to) {}
    range(int from, int to, int step) : _begin(from, step), _finish(to, step) {}
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