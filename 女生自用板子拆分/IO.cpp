#include "Headers.cpp"

// char buf[1<<23],*p1=buf,*p2=buf,obuf[1<<23],*O=obuf; // 或者用fread更难调的快读
// #define getchar() (p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21,stdin),p1==p2)?EOF:*p1++)

template <class T>
void print(T x)
{
    if (x < 0)
    {
        x = -x;
        putchar('-');
        // *O++ = '-';
    }
    if (x > 9)
        print(x / 10);
    putchar(x % 10 + '0');
    // *O++ = x%10 + '0'
}
// fwrite(obuf,O-obuf,1,stdout);

template <class T>
inline void qr(T &n)
{
    n = 0;
    int c = getchar();
    if (c == EOF)
        throw "End of file!";
    bool sgn = 0;

    while (!isdigit(c))
    {
        if (c == '-')
            sgn ^= 1;
        c = getchar();
        if (c == EOF)
            throw "End of file!";
    }

    while (isdigit(c))
    {
        n = (n * 10) + (c ^ 0x30);
        c = getchar();
    }

    if (sgn)
        n = -n;
}

inline char qrc()
{
    char c;
    while (!islower(c = getchar()))
        ;
    return c;
}