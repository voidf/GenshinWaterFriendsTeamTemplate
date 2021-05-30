# 字符串

## KMP
```cpp
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
```

## EXKMP
```cpp
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
```

## 字典树
```cpp
#include <cstdio>
const int N = 500010;
char s[60];
int n, m, ch[N][26], tag[N], tot = 1;
int main() {
	scanf("%d", &n);
	for (int i = 1; i <= n; ++i) {
		scanf("%s", s + 1);
		int u = 1;
		for (int j = 1; s[j]; ++j) {
			int c = s[j] - 'a';
			if (!ch[u][c]) ch[u][c] = ++tot;
			u = ch[u][c];
		}
		tag[u] = 1;
	}
	scanf("%d", &m);
	while (m--) {
		scanf("%s", s + 1);
		int u = 1;
			for (int j = 1; s[j]; ++j) {
				int c = s[j] - 'a';
				u = ch[u][c];
				if (!u) break; // 不存在对应字符的出边说明名字不存在
			}
		if (tag[u] == 1) {
			tag[u] = 2;
			puts("OK");
		}
		else if (tag[u] == 2)
			puts("REPEAT");
		else
			puts("WRONG");
	}
	return 0;
}
```

## AC机

```cpp
#define Aho_CorasickAutomaton 2000010
#define CharacterCount 26
struct TrieNode
{
    TrieNode *son[CharacterCount], *fail;
    // LL word_count;
    LL logs;

} T[Aho_CorasickAutomaton];
vector<TrieNode *> FailEdge[Aho_CorasickAutomaton];
LL AC_counter = 0;

vector<TrieNode *> trieIndex;

TrieNode *insertWords(string &s)
{
    auto root = &T[0];
    for (auto i : s)
    {
        auto nxt = i - 'a';
        if (root->son[nxt] == NULL)
            root->son[nxt] = &T[++AC_counter];
        root = root->son[nxt];
    }
    // word_count[root]++;

    return root; // 返回含单词的节点号
} // 用例：trieIndex.push_back(insertWords(s));

TrieNode *insertWords(char *s, LL &sLen)
{
    auto root = &T[0];
    for (auto i = 0; i < sLen; i++)
    {
        auto nxt = s[i] - 'a';
        if (root->son[nxt] == NULL)
            root->son[nxt] = &T[++AC_counter];
        root = root->son[nxt];
    }
    // word_count[root]++;

    return root; // 返回含单词的节点号
}

void getFail()
{
    queue<TrieNode *> Q; // bfs用
    for (auto i = 0; i < CharacterCount; i++)
    {
        if (T[0].son[i] != NULL)
        {
            T[0].son[i]->fail = &T[0];
            Q.push(T[0].son[i]);
        }
    }
    while (!Q.empty())
    {
        auto now = Q.front();
        Q.pop();
        now->fail = now->fail == NULL ? &T[0] : now->fail;
        for (auto i = 0; i < CharacterCount; i++)
        {
            if (now->son[i] != NULL)
            {
                now->son[i]->fail = now->fail->son[i];
                Q.push(now->son[i]);
            }
            else
            {
                now->son[i] = now->fail->son[i];
            }
        }
    }
} // 先设T[0].fail=0;所有单词插完以后调用一次

LL query(string &s)
{
    auto now = &T[0];
    auto ans = 0;
    for (auto i : s)
    {
        now = now->son[i - 'a'];
        now = now == NULL ? &T[0] : now;
        now->logs++;
        // for (auto j = now; j /*&& ~word_count[j]*/; j = fail[j])
        // {
        //     // ans += word_count[j];
        //     // cout << "j:" << j << endl;
        //     // if (word_count[j])
        //     logs[j]++;
        //     // for (auto k : word_position[j])
        //     //     pattern_count[k]++;
        //     // word_count[j] = -1; // 标记已经遍历的节点
        // }
    }

    for (auto i = 1; i <= AC_counter; i++)
    {
        FailEdge[T[i].fail - T].push_back(&T[i]);
    }

    return ans;
} // 查询母串，getFail后使用一次

LL query(char *s, LL &sLen)
{
    auto now = &T[0];
    auto ans = 0;
    for (auto i = 0; i < sLen; i++)
    {
        now = now->son[s[i] - 'a'];
        now = now == NULL ? &T[0] : now;
        now->logs++;
        // for (auto j = now; j /*&& ~word_count[j]*/; j = fail[j])
        // {
        // ans += word_count[j];
        // cout << "j:" << j << endl;
        // if (word_count[j])

        // for (auto k : word_position[j])
        //     pattern_count[k]++;
        // word_count[j] = -1; // 标记已经遍历的节点
        // }
    }

    for (auto i = 1; i <= AC_counter; i++)
    {
        FailEdge[T[i].fail - T].push_back(&T[i]);
    }

    return ans;
}

void AC_dfs(TrieNode *u)
{

    for (auto i : FailEdge[u - T])
    {
        AC_dfs(i);
        u->logs += i->logs;
    }
} // query完后使用，一般搜0号点

// 输出答案使用for(auto i:trieIndex)cout<<i.logs<<endl;这样

```

## 后缀数组
```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int N = 1000010;
char s[N];
int n, sa[N], rk[N << 1], oldrk[N << 1], id[N], cnt[N];
int main() {
	int i, m, p, w;
	scanf("%s", s + 1);
	n = strlen(s + 1);
	m = max(n, 300);
	for (i = 1; i <= n; ++i) ++cnt[rk[i] = s[i]];
	for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
	for (i = n; i >= 1; --i) sa[cnt[rk[i]]--] = i;
	for (w = 1; w < n; w <<= 1) {
		memset(cnt, 0, sizeof(cnt));
		for (i = 1; i <= n; ++i) id[i] = sa[i];
		for (i = 1; i <= n; ++i) ++cnt[rk[id[i] + w]];
		for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
			for (i = n; i >= 1; --i) sa[cnt[rk[id[i] + w]]--] = id[i];
		memset(cnt, 0, sizeof(cnt));
		for (i = 1; i <= n; ++i) id[i] = sa[i];
		for (i = 1; i <= n; ++i) ++cnt[rk[id[i]]];
		for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
		for (i = n; i >= 1; --i) sa[cnt[rk[id[i]]]--] = id[i];
		memcpy(oldrk, rk, sizeof(rk));
		for (p = 0, i = 1; i <= n; ++i) {
			if (oldrk[sa[i]] == oldrk[sa[i - 1]] &&
				oldrk[sa[i] + w] == oldrk[sa[i - 1] + w]) {
				rk[sa[i]] = p;
			}
			else {
				rk[sa[i]] = ++p;
			}
		}
	}
```

## 后缀自动机//各种应用待补
```cpp
struct state {
	int len, link;
	std::map<char, int> next;//可以考虑少一个log
};
// SAM 本身将会存储在一个 state 结构体数组中。我们记录当前自动机的大小 sz 和变量 last ，当前整个字符串对应的状态。
const int MAXLEN = 100000;
state st[MAXLEN * 2];
int sz, last;
//我们定义一个函数来初始化 SAM（创建一个只有初始状态的 SAM）。
void sam_init() {
	st[0].len = 0;
	st[0].link = -1;
	sz++;
	last = 0;
}
//最终我们给出主函数的实现：给当前行末增加一个字符，对应地在之前的基础上建造自动机。
void sam_extend(char c) {
	int cur = sz++;
	st[cur].len = st[last].len + 1;
	int p = last;
	while (p != -1 && !st[p].next.count(c)) {
		st[p].next[c] = cur;
		p = st[p].link;
	}
	if (p == -1) {
		st[cur].link = 0;
	}
	else {
		int q = st[p].next[c];
		if (st[p].len + 1 == st[q].len) {
			st[cur].link = q;
		}
		else {
			int clone = sz++;
			st[clone].len = st[p].len + 1;
			st[clone].next = st[q].next;
			st[clone].link = st[q].link;
			while (p != -1 && st[p].next[c] == q) {
				st[p].next[c] = clone;
				p = st[p].link;
			}
			st[q].link = st[cur].link = clone;
		}
	}
	last = cur;
}
```

## 回文树
```cpp
const LL M = 3e5 + 10;

struct PalindromicTreeNode
{
    LL son[26];
    LL suffix;
    LL curlen;
    LL cnt;
    char c;
} PTN[M];
// char orginalString[M];
LL PTNSIZE = 1; // SIZE - 1 actually
LL last = 0;

void __init__()
{
    PTN[0].curlen = 0;
    PTN[0].suffix = 1;
    PTN[0].c = '^';
    PTN[1].c = '#';
    PTN[1].curlen = -1;
}

LL __find__(LL pattern)
{
    while (PTN[PTNSIZE - PTN[pattern].curlen - 1].c != PTN[PTNSIZE].c)
        pattern = PTN[pattern].suffix;
    return pattern;
}

void __add__(char element)
{
    PTNSIZE++;
    PTN[PTNSIZE].c = element;
    LL offset = element - 97;
    LL cur = __find__(last);       // 可以加回文的点
    if (PTN[cur].son[offset] == 0) // 没前向边这条边
    {
        PTN[PTNSIZE].suffix = PTN[__find__(PTN[cur].suffix)].son[offset]; // 正在插入的字母的后缀边不可能是cur，所以要用chk往下找合法的
        PTN[cur].son[offset] = PTNSIZE;                                   // 这才是加前向边
        PTN[PTNSIZE].curlen = PTN[cur].curlen + 2;
    }
    last = PTN[cur].son[offset]; // 加过边以后last就是PTNSIZE
    PTN[last].cnt++;
}

LL __count__()
{
    LL re = 0;
    for (LL i = PTNSIZE; i >= 0; i--)
    {
        PTN[PTN[i].suffix].cnt += PTN[i].cnt; // 后缀边连接的节点走过次数要加上前面更高级的回文串节点走过次数
        re = max(re, PTN[i].curlen);          // 统计最长回文串长度
    }
    return re;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    __init__();
    string ss;
    cin >> ss;
    for (auto s : ss)
        __add__(s);
    __count__();
    LL ans = 0;
    for (auto i = 2; i <= PTNSIZE; i++)
    {
        ans = max(ans, PTN[i].cnt * PTN[i].curlen); // 最长回文子串
    }
    cout << ans << '\n';
    return 0;
}
```


# 附：快速IO

```cpp
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
    register char c = getchar();
    LL sgn = 1;

    while (c > '9' || c < '0')
    {
        if (c == '-')
            sgn = -1;
        c = getchar();
    }

    while (c <= '9' && c >= '0')
    {
        n = (n << 3) + (n << 1) + (c ^ 0x30);
        c = getchar();
    }

    n *= sgn;
}

inline char qrc()
{
    register char c = getchar();
    while (c < 'a' || c > 'z')
        c = getchar();
    return c;
}
```

# 附：没用的优化

```cpp
#pragma GCC optimize(3)
#pragma GCC target("avx")

```