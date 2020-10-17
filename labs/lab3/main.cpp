#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

using ll = long long;
using ld = long double;
const ll MAXX = 1e15;
const ll MOD = 1e9 + 7;

#define P pair
#define F first
#define S second
#define vec vector
#define pb push_back

struct pnt{
    int x, y;
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    string input, output;
    cin >> input >> output;
    int nc, np;
    cin >> nc;
    // ps[i-th class][j-th pixel in i-th class] = {y, x}
    vector<vector<pnt>> ps(nc);
    vector<int> avg(nc);
    for (int i = 0; i < nc; ++i) {
        cin >> np;
        ps[i].resize(np);
        for (int j = 0; j < np; ++j) {
            cin >> ps[i][j].y >> ps[i][j].x;
        }
    }

    ifstream fin(input, ios::binary);
    int n, m, x;
    fin >> n >> m;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fin >> x;
            cout << x;
        }
    }



    return 0;
}

