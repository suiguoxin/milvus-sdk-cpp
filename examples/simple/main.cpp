// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <random>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <set>

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"

const int topk = 50;
const int nlist = 1024;
const int search_limit = 512;
const int SIZE_COLLECTION = 46201;
const int SIZE_QUERY = 5133;
const std::string img_collection_name = "recipe_img_normalized";
const std::string ins_collection_name = "recipe_instr_normalized";
std::string outputResult = "/result/";
int m = 1024;


std::map<float,float> RESULT;

void CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

bool NRA(std::vector<std::vector<float> > ids, std::vector<std::vector<float> > scores, std::map<float,float> &RESULT)
{
    float minScore = -1.0;
    bool flag = false;
    std::unordered_map<float,float> UB, LB;
    std::vector<std::unordered_map<float,float> > matrix;
    std::unordered_set<float> seenIDSet;
    matrix.resize(ids.size());
    for (int i = 0; i < ids[0].size(); i++) {

        for (int j = 0; j < ids.size(); j++) {
            matrix[j][ids[j][i]] = scores[j][i];
            seenIDSet.insert(ids[j][i]);
        }
        
        for (auto id : seenIDSet) {
            LB [id] = 0;
            UB [id] = 0;
            for (int k = 0; k < ids.size(); k++) {
                if (matrix[k].find(id) == matrix[k].end()) {
                    LB[id] += minScore;
                    UB[id] += scores[k][i];
                } else {
                    LB[id] += matrix[k][id];
                    UB[id] += matrix[k][id];
                }
            }
        }
        if (LB.size() > topk) {
            std::vector<float> LB_value;
            for (auto x : LB)
                LB_value.push_back(x.second);
            int Rank = LB_value.size() - topk;
            std::nth_element(LB_value.begin(), LB_value.begin() + Rank, LB_value.end());
            float LB_topk = LB_value[Rank], UB_max = -100;

            for (auto x : LB) {
                float id = x.first, lbValue = x.second;
                if (lbValue <= LB_topk)
                    UB_max = std::max(UB_max, UB[id]);
            }
            
            if (LB_topk >= UB_max) {
                flag = true;
                break;
            }
        }
    }
    RESULT.clear();
    int validnumber = 0;
    for (auto x : LB) {
        float id = x.first, lb_value = x.second;
        RESULT[- lb_value] = x.first;
        while(RESULT.size() > topk) {
            auto tmp = RESULT.end();
            tmp--;
            RESULT.erase(tmp);
        }
    }
    return flag;
}

int main(int argc, char* argv[]) {
    std::string img_filename = "/embeddings/img_embeds_query.tsv";
    std::string ins_filename = "/embeddings/rec_embeds_query.tsv";
    printf("Experiments start...\n");
    
    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;


    status = client->LoadCollection(img_collection_name);
    CheckStatus("Failed to load collection:", status);
    status = client->LoadCollection(ins_collection_name);
    CheckStatus("Failed to load collection:", status);
    std::cout << "Load collection succesfully." << std::endl;


    milvus::CollectionStat coll_stat;
    status = client->GetCollectionStatistics(img_collection_name, coll_stat);
    CheckStatus("Failed to get collection statistics:", status);
    std::cout << "Collection " << img_collection_name << " row count: " << coll_stat.RowCount() << std::endl;
    status = client->GetCollectionStatistics(ins_collection_name, coll_stat);
    CheckStatus("Failed to get collection statistics:", status);
    std::cout << "Collection " << ins_collection_name << " row count: " << coll_stat.RowCount() << std::endl;
    
    
	std::ifstream img_in(img_filename);
    std::ifstream ins_in(ins_filename);

    std::vector<float> img_query_id, ins_query_id;
    std::vector<std::vector<float> > img_query, ins_query;
    img_query.resize(SIZE_QUERY);
    ins_query.resize(SIZE_QUERY);
    std::string line;
    int num = 0;
    for (; getline(img_in, line);) {
        img_query_id.push_back(std::stof(line.substr(0, line.find("\t"))));

		line.erase(0, line.find("[") + 1);
        for (int j = 0; j < m - 1; j++) {
            img_query[num].push_back(std::stof(line.substr(0, line.find(","))));
            line.erase(0, line.find(",") + 2);
        }
        img_query[num].push_back(std::stof(line.substr(0, line.find("]"))));
        //normalization;
        double sum = 0.0;
        for (auto i = 0; i < img_query[num].size(); i++)
            sum += img_query[num][i] * img_query[num][i];
        sum = sqrt(sum);
        for (auto i = 0; i < img_query[num].size(); i++)
            img_query[num][i] /= sum;
        num++;
    }
    std::cout << num << " img queries has been read." << std::endl;
    num = 0;
    for (; getline(ins_in, line);) {
        ins_query_id.push_back(std::stof(line.substr(0, line.find("\t"))));

		line.erase(0, line.find("[") + 1);
        for (int j = 0; j < m - 1; j++) {
            ins_query[num].push_back(std::stof(line.substr(0, line.find(","))));
            line.erase(0, line.find(",") + 2);
        }
        ins_query[num].push_back(std::stof(line.substr(0, line.find("]"))));
        //normalization;
        double sum = 0.0;
        for (auto i = 0; i < ins_query[num].size(); i++)
            sum += ins_query[num][i] * ins_query[num][i];
        sum = sqrt(sum);
        for (auto i = 0; i < ins_query[num].size(); i++)
            ins_query[num][i] /= sum;
        num++;
    }
    std::cout << num << " ins queries has been read." << std::endl;

    std::string output_result_path = outputResult + "qrels.txt";
    std::string output_lantency_path = outputResult + "latency.txt";
    std::ofstream out1(output_result_path);
    std::ofstream out2(output_lantency_path);

    for (int i = 0; i < SIZE_QUERY; i++)
    {
        double Begin_time = clock();
        int limit = 1;
        std::cout << "search start." << std::endl;
        while(limit < topk) limit *= 2;
        while(limit <= search_limit && limit <= SIZE_COLLECTION) {
            bool flag = false;
            int TOPK = limit;
            int nprobe = std::ceil(TOPK / (SIZE_COLLECTION / nlist));
            

            milvus::SearchArguments img_arguments{};
            img_arguments.SetCollectionName(img_collection_name);
            img_arguments.SetTopK(TOPK);
            img_arguments.AddTargetVector("img_embeds", img_query[i]);
            img_arguments.AddExtraParam("nprobe", nprobe);
            img_arguments.SetMetricType(milvus::MetricType::IP);
            milvus::SearchResults img_search_results{};
            auto status = client->Search(img_arguments, img_search_results);
            CheckStatus("Failed to search:", status);

            milvus::SearchArguments ins_arguments{};
            ins_arguments.SetCollectionName(ins_collection_name);
            ins_arguments.SetTopK(TOPK);
            ins_arguments.AddTargetVector("rec_embeds", ins_query[i]);
            ins_arguments.AddExtraParam("nprobe", nprobe);
            ins_arguments.SetMetricType(milvus::MetricType::IP);
            milvus::SearchResults ins_search_results{};
            status = client->Search(ins_arguments, ins_search_results);
            CheckStatus("Failed to search:", status);

            for (auto& img_result : img_search_results.Results()) {
                auto& img_ids = img_result.Ids().IntIDArray();
                auto& img_distances = img_result.Scores();
                if (img_ids.size() != img_distances.size()) {
                    std::cout << "img Illegal result!" << std::endl;
                    continue;
                }
                std::cout << "img Successfully search, " << img_ids.size() << "results." <<std::endl;
                for (auto& ins_result : ins_search_results.Results()) {
                    auto& ins_ids = ins_result.Ids().IntIDArray();
                    auto& ins_distances = ins_result.Scores();
                    if (ins_ids.size() != ins_distances.size()) {
                        std::cout << "img Illegal result!" << std::endl;
                        continue;
                    }
                    std::cout << "ins Successfully search, " << ins_ids.size() << "results." <<std::endl;

                    std::vector<std::vector<float> > ids, distances;
                    ids.resize(2);
                    distances.resize(2);
                    for (int i = 0; i < img_ids.size(); i++)
                        ids[0].push_back(img_ids[i]),
                        ids[1].push_back(ins_ids[i]),
                        distances[0].push_back(img_distances[i]),
                        distances[1].push_back(ins_distances[i]);

                    flag = NRA(ids, distances, RESULT);
                    std::cout << "NRA finish, has " << RESULT.size() << "results." << std::endl;
                }
            }

            if (flag) break;
            limit *= 2;
        }
        double End_time = clock();
        int rank = 0;
        for (auto x : RESULT) {
            out1 << img_query_id[i] << " " << x.second << " " << ++rank << " " << -x.first << std::endl;
            if (rank == 50) break;
        }
        out2 << img_query_id[i] << " " << (End_time - Begin_time) / CLOCKS_PER_SEC << std::endl;
        for (int ii = 0; ii < 50 - RESULT.size(); ii++)
            out1 << img_query_id[i] << " " << -1 << " " << ++rank << " " << -1 << std::endl;
        int number = i + 1;
        if (number % 5 == 0)
            std::cout << number << " queries searched." << std::endl;
    }
    return 0;
}
