#include <iostream>
#include <cstring>
using namespace std;
// 在此处补充你的代码
class Array2{
    int r_size, c_size;
    int** ptr;
    public:
        Array2(int r=0, int c=0);
        ~Array2();

        int row_size(){return r_size;}
        int col_size(){return c_size;}
        
        int** get_ptr(){return ptr;}
        Array2 & operator=(const Array2 & a);
        int* & operator[](int i){ return ptr[i]; }

};
Array2::Array2(int r, int c):r_size(r),c_size(c){
    if(r==0 && c==0){
        ptr = NULL;
    }
    else{
        ptr = new int*[r];
        for(int i=0;i<r;i++){
            ptr[i] = new int[c];
        }
    }
}
Array2::~Array2(){
    if(ptr){
        for(int i=0;i<c_size;i++){
            delete [] ptr[i];
        delete [] ptr;
        }
    }
}
Array2 & Array2::operator=(const Array2 & a){
    
    //ptr = a.ptr
    if(ptr==a.ptr){return *this;}
    
    //a.ptr = NULL
    if(a.ptr==NULL){
        if(ptr){
            for(int i=0;i<c_size;i++){
                delete [] ptr[i];
            }
            delete [] ptr;
        }
        ptr = NULL;
        r_size=0;
        c_size=0;
        return *this;
    }
    //a.ptr != NULL
    if(r_size*c_size < a.r_size*a.c_size){
        if(ptr){
            for(int i=0;i<c_size;i++){
                delete [] ptr[i];
            }
            delete [] ptr;
        }
        ptr = new int*[a.r_size];
        for(int i=0;i<a.r_size;i++){
            ptr[i] = new int[a.c_size];
        }
    }
    r_size=a.r_size;
    c_size=a.c_size;
    memcpy(ptr,a.ptr,sizeof(int)*a.r_size*a.c_size);
    return *this;

}

int main() {
    Array2 a(3,4);
    int i,j;
    for(  i = 0;i < 3; ++i )
        for(  j = 0; j < 4; j ++ )
            a[i][j] = i * 4 + j;
    for(  i = 0;i < 3; ++i ) {
        for(  j = 0; j < 4; j ++ ) {
            cout << a[i][j] << ",";
        }
        cout << endl;
    }
    
    cout << "next" << endl;
    Array2 b;     b = a;
    for(  i = 0;i < 3; ++i ) {
        for(  j = 0; j < 4; j ++ ) {
            cout << b[i][j] << ",";
        }
        cout << endl;
    }
    return 0;
}