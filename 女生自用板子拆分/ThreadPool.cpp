// 助教写的线程池，附用例
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <functional>
#include <type_traits>
#include <future>
#include <memory>
#include <algorithm>

class thread_pool
{
public:
    explicit thread_pool(size_t size)
        : isDestroy(false)
    {
        for (size_t i = 0; i < size; i++)
        {
            _threads.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;
                    {
                        //  Get lock
                        std::unique_lock<std::mutex> lock(_mutex);
                        //  Wait task
                        _condition.wait(lock, [this]() { return this->isDestroy || !this->_tasks.empty(); });
                        //  Destroy thread
                        if (this->isDestroy && this->_tasks.empty())
                        {
                            return;
                        }
                        //  Get task
                        task = std::move(this->_tasks.front());
                        this->_tasks.pop();
                    }
                    //  Do task
                    task();
                }
            });
        }
    }

    ~thread_pool()
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            isDestroy = true;
        }
        //  Notify all
        _condition.notify_all();
        // Join all
        for_each(_threads.begin(), _threads.end(), std::mem_fn(&std::thread::join));
    }

    template <class F, class... Args>
    decltype(auto) submit(F &&f, Args &&...args);

    inline size_t getThreadSize() const
    {
        return _threads.size();
    }

    inline size_t getTaskSize() const
    {
        return _tasks.size();
    }

private:
    std::vector<std::thread> _threads;
    std::queue<std::function<void()>> _tasks;
    mutable std::mutex _mutex;
    std::condition_variable _condition;

    bool isDestroy;
};

template <class F, class... Args>
inline decltype(auto) thread_pool::submit(F &&f, Args &&...args)
{
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> result = task->get_future();

    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (isDestroy)
        {
            throw std::runtime_error("Add task on detroyed thread pool");
        }

        //  add task
        _tasks.emplace([task]() { (*task)(); });
    }
    _condition.notify_one();
    return result;
}

// !!!!!!!!!!!!!!!!!!用例!!!!!!!!!!!!!!!!!!!!
#include <unordered_set>
#include "foreach.cpp"
std::vector<int> P;

void prework()
{
	for (auto i : range(2, 40001))
	{
		bool f = 1;
		for (auto j : range(2, i))
		{
			if (i % j == 0)
			{
				f = 0;
				break;
			}
		}
		if (f)
			P.emplace_back(i);
	}
	std::cerr << "done with primes:" << P.size() << endl;
}

void solve(int l, int r)
{
	for (auto i : range(l, min((int)r, (int)P.size())))
	{
		for (auto j : range(i + 1, P.size()))
		{
			auto ki = P[i];
			auto ki2 = P[j];
			std::unordered_set<int> S;
			for (auto ni : range(1, 100001))
			{
				auto k1 = power(ni, ni, ki);
				auto k2 = power(ni, ni, ki2);
				auto kk = k1 * 40000 + k2;
				// cerr << kk << endl;
				auto sz = S.size();
				S.emplace(kk);
				if (S.size() == sz)
				{
					if (j % 1000 == 0)
					{
						std::cerr << "doing " << ki << " with " << ki2 << "..." << S.size() << "\n";
					}
					break;
				}
			}
			if (S.size() > 99900)
			{
				std::cerr << ki << " and " << ki2 << " might work." << S.size() << "\n";
			}
			if (S.size() == 100000)
			{
				std::cerr << ki << " and " << ki2 << " worked." << S.size() << "\n";
				return;
			}
		}
	}
}

signed main()
{
    int	T = 1;
	prework();
	// for (int i = 1; i <= T; i++)
	// {
	while (T--)
	// while (~scanf("%d%d%d", &L, &A, &N))
	{
		thread_pool tp(16);
		for (auto i : range(16))
		{
			tp.submit(solve, 265 * i, 265 * (i + 1));
		}
		// cout << "Case #" << i << ": ";
		// ~scanf("%d %d", &n, &k)
		// cin >> n >> k)
		// solve(2000, 4203);
	}

	return 0;
}