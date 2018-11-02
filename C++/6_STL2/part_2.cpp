#include <iostream>
#include <string>
using namespace std;
// 在此处补充你的代码
template <class T1>
class CMyistream_iterator{
    T1 var;
    istream& is;
    public:
        CMyistream_iterator(istream& cin): is(cin){
            is >> var;
        }
        T1 operator*(){
            return var;
        }
        void operator++(int){
            is >> var;
        }

};
int main()
{
    CMyistream_iterator<int> inputInt(cin);
    int n1,n2,n3;
    n1 = * inputInt; //读入 n1
    int tmp = * inputInt;
    cout << tmp << endl;
    inputInt ++;
    n2 = * inputInt; //读入 n2
    inputInt ++;
    n3 = * inputInt; //读入 n3
    cout << n1 << "," << n2<< "," << n3 << endl;
    CMyistream_iterator<string> inputStr(cin);
    string s1,s2;
    s1 = * inputStr;
    inputStr ++;
    s2 = * inputStr;
    cout << s1 << "," << s2 << endl;
    return 0;
}