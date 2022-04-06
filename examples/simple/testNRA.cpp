#include<bits/stdc++.h>
#include<map>

int main()
{
    std::map<float,float> RESULT;
    std::vector<float> img_ids = {1,2,3,4,5};
    std::vector<float> ins_ids = {2,3,1,4,5};
    std::vector<float> img_distances = {1,0.8,0.5,0.3,0.1};
    std::vector<float> ins_distances = {0.8,0.7,0.3,0.2,0.1};
    int topk=50;


    std::unordered_map<float,float> UB, LB;
    std::vector<std::unordered_map<float,float> > matrix;
    matrix.resize(2);
    for (int i = 0; i < ins_ids.size(); i++) {
        matrix[0][img_ids[i]] = img_distances[i];
        matrix[1][ins_ids[i]] = ins_distances[i];
        for (auto x : matrix[0])
            if (matrix[1].find(x.first) == matrix[1].end()) {
                LB[x.first] = x.second + min_value;
                UB[x.first] = x.second + distance[1][i];
            }
            else {
                LB[x.first] = x.second + matrix[1][x.first];
                UB[x.first] = LB[x.first];
            }
        for (auto x : matrix[1])
            if (matrix[0].find(x.first) == matrix[0].end()) {
                LB[x.first] = x.second + min_value;
                UB[x.first] = x.second + distance[0][i];
            }
        if (LB.size() > topk) {
            std::set<float> merge_score;
            for (auto x : LB) {
                merge_score.insert(x.second);
                while(merge_score.size() > topk) merge_score.erase(merge_score.begin());
            }
            float LB_topk = (*merge_score.begin()), UB_max = -100;
            for (auto x : LB)
                if (x.second < LB_topk) UB_max = std::max(UB_max, UB[x.first]);
            // if (LB_topk >= UB_max) break;
        }

        std::cout<<"LB"<<std::endl;
        for (auto x:LB)
            std::cout<<x.first<<" "<<x.second<<std::endl;
        std::cout<<"UB"<<std::endl;
        for (auto x:UB)
            std::cout<<x.first<<" "<<x.second<<std::endl;
        


    }
    RESULT.clear();
    for (auto x : LB) {
        float id = x.first;
        RESULT[- matrix[0][id] - matrix[1][id]] = x.first;
        while(RESULT.size() > topk) {
            auto tmp = RESULT.end();
            tmp--;
            RESULT.erase(tmp);
        }
    }




    int rank = 0;
    for (auto x : RESULT) {
    std::cout << " " << x.second << " " << ++rank << " " << -x.first << std::endl;
    if (rank == 50) break;
    }
}