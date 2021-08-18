#include <bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int, int>
const int maxn = 2e5 + 5;
int a[maxn];
int b[maxn];
int h;
int rxa(int x, int al, int ar)
{
    if (x <= al)
        x = 1;
    if (x > ar)
        x = ar;
    return x;
}
int rxb(int x, int al, int ar)
{
    if (x < al)
        x = al;
    if (x >= ar)
        x = h;
    return x;
}
signed main()
{
    ios::sync_with_stdio(false);
    /*freopen("1.in","r",stdin);
    freopen("1.out","w",stdout);*/
    int T;
    cin >> T;
    while (T--)
    {
        int n, s, t;
        cin >> n >> s >> t >> h;
        int suma = 0, sumb = 0;
        for (int i = 1; i <= n - 1; i++)
        {
            cin >> a[i];
            suma += a[i];
        }
        for (int i = 1; i <= n - 1; i++)
        {
            cin >> b[i];
            sumb += b[i];
        }
        sort(a + 1, a + n);
        sort(b + 1, b + n);
        int al, ar;
        int bl, br;
        for (int i = 0, j = n - 1; i < s; i++)
        {
            suma -= a[j - i];
            sumb -= b[j - i];
        }
        ar = (s == 0) ? h : a[n - s];
        al = (t == 0) ? 1 : a[t];
        br = (s == 0) ? h : b[n - s];
        bl = (t == 0) ? 1 : b[t];
        for (int i = 0, j = 1; i < t; i++)
        {
            suma -= a[j + i];
            sumb -= b[j + i];
        }
        int ned = sumb - suma;
        if (ar - bl <= ned)
        {
            cout << "IMPOSSIBLE\n";
        }
        else
        {
            int ans = max(-(h-1),ned + 1);
            //ans>ned
            if (al - bl > ned)
            {
                int l = 1;
                int r = rxb(al - ned - 1, bl, br);
                /*int r = al - ned - 1;
                if (r >= br)
                {
                    r = h;
                }*/
                ans = min(ans, 1 - r);
            }
            if (ar - br > ned)
            {
                int r = h;
                //int l = br + ned + 1;
                int l = rxa(br + ned + 1, al, ar);
                //if(l<)
                ans = min(ans, l-r);
            }
            cout << ans << "\n";
        }
    }
}