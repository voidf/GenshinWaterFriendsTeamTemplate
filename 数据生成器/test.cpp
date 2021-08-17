#include <bits/stdc++.h>
#define int long long
using namespace std;
typedef long long ll;

int n, s, t, h;
const int maxn = 1e5 + 10;
// int a[maxn];
// int b[maxn];
vector<int> a, b, aa, bb;

#define all(x) x.begin(), x.end()

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    // freopen("in.txt","r",stdin);
    // freopen("out2.txt","w",st)
    int T;
    cin >> T;
    a.clear();
    b.clear();
    //a.resize(maxn);
    //b.resize(maxn);
    while (T--)
    {
        int _;
        a.clear();
        b.clear();
        cin >> n >> s >> t >> h;
        for (int i = 0; i < n - 1; ++i)
            cin >> _, a.push_back(_);
        for (int i = 0; i < n - 1; ++i)
            cin >> _, b.push_back(_);
        int ans = 0x3f3f3f3f;

        for (int i = 1; i <= h; ++i)
        {
            aa = a;
            aa.push_back(i);
            sort(all(aa));
            int ansa = 0;
            for (int p = t; p < n - s; ++p)
            {
                ansa += aa[p];
            }
            // ansa=ansa/double(n-s-t);
            for (int j = 1; j <= h; ++j)
            {
                bb = b;
                bb.push_back(j);
                sort(all(bb));
                int ansb = 0;
                for (int p = t; p < n - s; ++p)
                {
                    ansb += bb[p];
                }
                if (ansa > ansb)
                {
                    ans = min(ans, i - j);
                }
            }
        }
        if (ans >= 0x3f3f3f3f)
        {
            cout << "IMPOSSIBLE\n";
        }
        else
            cout << ans << "\n";
    }

    return 0;
}
/*

3
3 1 1 4
1 3
2 4
4 1 1 9
4 4 5
4 5 5
4 1 1 9
4 5 5
4 4 5

*/