#include <iostream>
#include <string>
#include <list>
#include <vector>

using namespace std;
int main(){
    string cmd;
    string id;
    vector <list<int> > container;
    while(cin >> cmd){
        if(cmd == "new"){
            cin >> id;
            list<int> l;
            container.push_back(l);
        }
        else if(cmd == "add"){
            if(container.empty()) continue;
            int num;
            cin >> id >> num;
            container[stoi(id)-1].push_back(num);

        }
        else if(cmd == "merge"){
            if(container.empty()) continue;
            string id1,id2;
            cin >> id1 >> id2;
            container[stoi(id1)-1].splice(container[stoi(id1)-1].end(),container[stoi(id2)-1]);
            container[stoi(id1)-1].sort();
        }
        else if(cmd == "out"){
            if(container.empty()) continue;
            cin >> id;
            list<int> l = container[stoi(id)-1];
            list<int>::iterator it;
            for(it=l.begin(); it != l.end(); it++){
                cout << *it << " ";
            }
            cout << endl;

        }
        else if(cmd == "unique"){
            if(container.empty()) continue;
            cin >> id;
            container[stoi(id)-1].unique();
        }
        else cout << "Import Error!!";

        
    }

}