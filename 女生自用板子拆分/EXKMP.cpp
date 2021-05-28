#include "Headers.cpp"

#define KMPM 114514
inline void Getnext(LL next[], char t[])
{
    LL p1 = 0;
    LL p2 = next[0] = -1;
    LL strlen_t = strlen(t);
    while (p1 < strlen_t)
    {
        if (p2 == -1 || t[p1] == t[p2])
            next[++p1] = ++p2;
        else
            p2 = next[p2];
    }
}

inline void KMP(char string[], char pattern[], LL next[])
{
    LL p1 = 0;
    LL p2 = 0;
    LL strlen_string = strlen(string);
    LL strlen_pattern = strlen(pattern);
    while (p1 < strlen_string)
    {
        if (p2 == -1 || string[p1] == pattern[p2])
            p1++, p2++;
        else
            p2 = next[p2];
        if (p2 == strlen_pattern)
            printf("%lld\n", p1 - strlen_pattern + 1), p2 = next[p2];
    }
}
#define EXKMPM 114514

string pattern;
string s;
LL nxt[EXKMPM];
LL extend[EXKMPM];

void getNEXT(string &pattern, LL next[])
{
    LL pLen = pattern.length();
    LL a = 0, k = 0;

    next[0] = pLen;
    for (auto i = 1; i < pLen; i++)
    {
        if (i >= k || i + next[i - a] >= k)
        {
            if (i >= k)
                k = i;
            while (k < pLen && pattern[k] == pattern[k - i])
                k++;
            next[i] = k - i;
            a = i;
        }
        else
        {
            next[i] = next[i - a];
        }
    }
}

void EXKMP(string &s, string &pattern, LL extend[], LL next[]) // string类得配O2不然过不了
{
    LL pLen = pattern.length();
    LL sLen = s.length();
    LL a = 0, k = 0;

    getNEXT(pattern, next);

    for (auto i = 0; i < sLen; i++)
    {
        if (i >= k || i + next[i - a] >= k)
        {
            if (i >= k)
                k = i;
            while (k < sLen && k - i < pLen && s[k] == pattern[k - i])
                k++;
            extend[i] = k - i;
            a = i;
        }
        else
        {
            extend[i] = next[i - a];
        }
    }
}
