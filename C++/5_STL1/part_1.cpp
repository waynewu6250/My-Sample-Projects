#include <iostream> 
#include <iterator> 
#include <set> 
using namespace std;
int main() { 
    int a[] = {8,7,8,9,6,2,1}; 
// 在此处补充你的代码
    set <int> v;
    int len = sizeof(a)/sizeof(a[0]);
    for(int i=0;i<len;i++) v.insert(a[i]);

    ostream_iterator<int> o(cout," ");
    copy( v.begin(),v.end(),o); 
    cout << endl;
    return 0;
}