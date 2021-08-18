/*
#include <windows.h>
#include <bits/stdc++.h>

using namespace std;
int main()
{
    system("g++ data.cpp -o data --std=c++17");
    system("g++ std.cpp -o std --std=c++17");
    system("g++ test.cpp -o test --std=c++17");
    int t = 10000;
    while (t)
    {
        t--;
        
        system("data.exe > data.txt");
        clock_t st=clock();
        system("std.exe < data.txt > std.txt");
        clock_t end=clock();
        system("test.exe < data.txt > test.txt");
        if (system("fc std.txt test.txt"))
            break;
        cout<<"TIME: "<<end-st<<" ms\n\n";
    }
    if (t == 0)
        cout << "Accepted!" << endl;
    else
        cout << "Wrong Answer!" << endl;
    return 0;
}
*/

//重点！数据比较器
#include <bits/stdc++.h>
using namespace std;
int main()
{
    system("g++ ./data.cpp -o data --std=c++17");
    system("g++ ./std.cpp -o std --std=c++17");
    system("g++ ./test.cpp -o test --std=c++17");
    int t = 10000;
    while (t--)
    {
        system("./data.exe > ./data.txt");
        clock_t st = clock();
        system("./std.exe < ./data.txt > ./std.txt");
        clock_t end = clock();
        system("./test.exe < ./data.txt > ./test.txt");
        if (system("diff ./std.txt ./test.txt"))
            break;
        cout << "TIME: " << end - st << " ms\n\n";
    }
    if (t == 0)
        cout << "Accepted!" << endl;
    else
        cout << "Wrong Answer!" << endl;
    return 0;
}
