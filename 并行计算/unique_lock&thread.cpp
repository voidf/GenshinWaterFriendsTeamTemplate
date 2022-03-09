#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;

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

void fun()
{
    {
        std::unique_lock<std::mutex> lck(mtx);
        std::cerr << std::this_thread::get_id() << std::endl;
    }
}

int main()
{
    std::vector<std::thread> V;
    for (auto i : range(4))
    {
        V.emplace_back(fun);
    }
    for (auto &i : V)
        i.join();

    return 0;
}