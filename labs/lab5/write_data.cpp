#include <stdio.h>
#include <cstdlib>
#include <ctime>

using namespace std;

const int SZ = 8e6;

int main(int argc, char *argv[]) {

    srand((unsigned) time(0));

    int sz = SZ;
    fwrite(&sz, sizeof(int), 1, stdout);
    int randomNumber;
    for (int index = 0; index < sz; index++) {
        randomNumber = (rand() % SZ) + 1;
        fwrite(&randomNumber, sizeof(int), 1, stdout);
    }

    return 0;
}

