#include<bits/stdc++.h>
#define random(a, b) rand()%(b-a+1) + a
using namespace std;

void creatData(int n, string filename) {
	fstream file(filename.c_str(), ios::out);
	int *tree = new int [n];
	for (int i = 0; i < n; ++i) {
		tree[i] = i + 1;
	}
	int root = random(0, n - 1);
	swap(tree[root], tree[n - 1]);
	int nxt_idx = n - 2;
	queue<int> Que;
	file << n << endl;
    // Ëæ»úÈ¨Öµ
	for (int i = 0; i < n; ++i) {
		file << random(-1024, 1024) << ' ';
	}
	file << endl;
	Que.push(tree[n - 1]);
	while (!Que.empty()) {
		int now = Que.front();
		Que.pop();
		int cnt = random(1, 3);
		for (int i = 0; i < cnt; ++i) {
			int tmp_idx = random(0, nxt_idx); 
			swap(tree[tmp_idx], tree[nxt_idx]);
			file << now << ' ' << tree[nxt_idx] << endl;
			Que.push(tree[nxt_idx]);
			nxt_idx--;
			if (nxt_idx == -1) break;
		}
		if (nxt_idx == -1) break;
	}
}

int main()
{
    srand(time(0));
	creatData(10, "creatTree.txt");
	// creatData(10, "creatTree10.txt");
}