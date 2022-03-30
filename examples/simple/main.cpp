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

bool Search_topk(int TOPK, std::vector<float> img_target_vector, std::vector<float> ins_target_vector) {
    
}

int main(int argc, char* argv[]) {
    std::string img_filename = "/embeddings/img_embeds_query.tsv";
    std::string ins_filename = "/embeddings/rec_embeds_query.tsv";
    printf("Experiments start...\n");
    
    auto img_client = milvus::MilvusClient::Create();
    auto ins_client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = img_client->Connect(connect_param);
    CheckStatus("Failed to connect milvus server:", status);
    status = ins_client->Connect(connect_param);
    CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;


    status = img_client->LoadCollection(img_collection_name);
    CheckStatus("Failed to load collection:", status);
    status = ins_client->LoadCollection(ins_collection_name);
    CheckStatus("Failed to load collection:", status);
    
	std::ifstream img_in(img_filename.c_str());
    if (!img_in)
	{
		std::cout << "Fail to read img query " << std::endl;
		return 0;
	}
    std::ifstream ins_in(ins_filename.c_str());
    if (!img_in)
	{
		std::cout << "Fail to read ins query " << std::endl;
		return 0;
	}
    std::vector<float> img_query_id, ins_query_id;
    std::vector<std::vector<float> > img_query, ins_query;
    img_query.resize(SIZE_QUERY);
    ins_query.resize(SIZE_QUERY);
    std::string line;
    int num = 0;
    for (; getline(img_in, line);)
	{
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
    num = 0;
    for (; getline(ins_in, line);)
	{
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

    std::string output_result_path = outputResult + "qrels.txt";
    std::string output_lantency_path = outputResult + "latency.txt";
    std::ofstream out1(output_result_path.c_str());
    if (!out1.is_open())
    {
        std::cout << "Cannot open file out1" << std::endl;
        return 0;
    }
    std::ofstream out2(output_lantency_path.c_str());
    if (!out2.is_open())
    {
        std::cout << "Cannot open file out2" << std::endl;
        return 0;
    }

    for (int i = 0; i <= SIZE_QUERY; i++)
    {
        double Begin_time = clock();
        int limit = 1;
        while(limit < topk) limit *= 2;
        while(limit <= search_limit && limit <= SIZE_COLLECTION) {
            bool flag;
            int TOPK = limit;
            int nprobe = TOPK / (SIZE_COLLECTION / nlist);
            
            milvus::SearchArguments img_arguments{};
            img_arguments.SetCollectionName(img_collection_name);
            img_arguments.SetTopK(TOPK);
            img_arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());
            img_arguments.AddTargetVector("img_embeds", img_query[i]);
            img_arguments.AddExtraParam("nprobe", nprobe);
            milvus::SearchResults img_search_results{};
            auto status = img_client->Search(img_arguments, img_search_results);
            CheckStatus("Failed to search:", status);
            std::cout << "img Successfully search." << std::endl;

            milvus::SearchArguments ins_arguments{};
            ins_arguments.SetCollectionName(ins_collection_name);
            ins_arguments.SetTopK(TOPK);
            ins_arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());
            ins_arguments.AddTargetVector("rec_embeds", ins_query[i]);
            ins_arguments.AddExtraParam("nprobe", nprobe);
            milvus::SearchResults ins_search_results{};
            status = ins_client->Search(ins_arguments, ins_search_results);
            CheckStatus("Failed to search:", status);
            std::cout << "ins Successfully search." << std::endl;

            for (auto& img_result : img_search_results.Results()) {
                auto& img_ids = img_result.Ids().IntIDArray();
                auto& img_distances = img_result.Scores();
                if (img_ids.size() != img_distances.size()) {
                    std::cout << "img Illegal result!" << std::endl;
                    continue;
                }
                for (auto& ins_result : ins_search_results.Results()) {
                    auto& ins_ids = ins_result.Ids().IntIDArray();
                    auto& ins_distances = ins_result.Scores();
                    if (ins_ids.size() != ins_distances.size()) {
                        std::cout << "img Illegal result!" << std::endl;
                        continue;
                    }

                    std::unordered_map<float,float> merge_score;
                    RESULT.clear();
                    for (int i = 0; i < ins_ids.size(); i++){
                        if (merge_score.find(img_ids[i]) == merge_score.end())
                            merge_score[img_ids[i]] = img_distances[i];
                        else RESULT[img_ids[i]] = merge_score[img_ids[i]] + img_distances[i];
                        if (merge_score.find(ins_ids[i]) == merge_score.end())
                            merge_score[ins_ids[i]] = ins_distances[i];
                        else RESULT[ins_ids[i]] = merge_score[ins_ids[i]] + ins_distances[i];
                        if (RESULT.size() > topk)
                        {
                            auto tmp = RESULT.end();
                            tmp--;
                            RESULT.erase(tmp);
                        }
                    }
                    if (RESULT.size() == topk) flag = true;
                    flag = false;
                    
                }
            }




            if (flag) {
                double End_time = clock();
                for (auto x : RESULT)
                    out1 << img_query_id[i] << x.first << x.second << std::endl;
                out2 << img_query_id[i] << End_time - Begin_time << std::endl;
                break;
            }
            limit *= 2;
        }
        int number = i + 1;
        if (number % 5 == 0)
            std::cout << number << "queries searched." << std::endl;
    }
    return 0;
}
