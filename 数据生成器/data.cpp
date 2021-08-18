#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

signed main()
{
    // freopen("in.txt","w",stdout);
    ios::sync_with_stdio(false);
    cin.tie(0);
    srand(time(0));
    int T=rand()%20+1;
    cout<<T<<"\n";
    int a[1010],b[1010];
    while(T--){
        int n,s,t,h;
        h=rand()%100+1;
        n=rand()%10+1;
        do{
            s=rand()%11;
            t=rand()%11;
        }while(s>=n || t>=n || s+t>n-1);
        for(int i=0;i<n-1;++i){
            a[i]=rand()%h+1;
            b[i]=rand()%h+1;
        }
        cout<<n<<" "<<s<<" "<<t<<" "<<h<<"\n"; 
        for(int i=0;i<n-1;++i) cout<<a[i]<<" ";
        cout<<"\n";
        for(int i=0;i<n-1;++i) cout<<b[i]<<" ";
        cout<<"\n";
    }
    return 0;
}