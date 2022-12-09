#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <chrono>
#include <vector>

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"

int topk = 50;
// int ef = 64;
const int SIZE_QUERY = 10000;
const std::string img_collection_name = "recipe_img";
const std::string ins_collection_name = "recipe_instr";
std::string img_filename = "/embeddings/img_embeds.tsv";
// std::string filter_filename = "/embeddings/price.tsv";
std::string outputResult = "/result/";
int m = 1024;

std::map<float, float> RESULT;

void
CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

void search(int ef, int price){
    printf("Search starting...\n");

    auto client = milvus::MilvusClient::Create();
    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    status = client->LoadCollection(img_collection_name);
    CheckStatus("Failed to load collection:", status);
    std::cout << "Load collection succesfully." << std::endl;

    milvus::CollectionStat coll_stat;
    status = client->GetCollectionStatistics(img_collection_name, coll_stat);
    CheckStatus("Failed to get collection statistics:", status);
    std::cout << "Collection " << img_collection_name << " row count: " << coll_stat.RowCount() << std::endl;
    
    std::ifstream img_in(img_filename);
    // std::ifstream filter_in(filter_filename);

    std::vector<float> img_query_id;
    // std::vector<int> prices;
    std::vector<std::vector<float> > img_query;
    img_query.resize(SIZE_QUERY);
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
        // normalization;
        double sum = 0.0;
        for (auto i = 0; i < img_query[num].size(); i++) sum += img_query[num][i] * img_query[num][i];
        sum = sqrt(sum);
        for (auto i = 0; i < img_query[num].size(); i++) img_query[num][i] /= sum;
        num++;
    }
    std::cout << num << " img queries has been read." << std::endl;
    
    // num = 0;
    // for (; getline(filter_in, line);) {
    //     line.erase(0, line.find("\t") + 1);
    //     // ingre_num.push_back(std::stoi(line.substr(0, line.find("\t"))));
    //     // line.erase(0, line.find("\t") + 1);
    //     prices.push_back(std::stoi(line.substr(0, line.find("\n"))));
    //     num++;
    // }
    // std::cout << num << " filter has been read." << std::endl;
    std::string output_result_path = outputResult + "qrels-" + std::to_string(price) + "-" + std::to_string(ef) + ".tsv";
    std::string output_lantency_path = outputResult + "latency-" + std::to_string(price) + "-" + std::to_string(ef) + ".tsv";
    std::ofstream out1(output_result_path);
    std::ofstream out2(output_lantency_path);

    for (int i = 0; i < SIZE_QUERY; i++) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        bool flag = false;
        int TOPK = topk;

        milvus::SearchArguments img_arguments{};
        img_arguments.SetCollectionName(img_collection_name);
        img_arguments.SetTopK(TOPK);
        img_arguments.AddTargetVector("img_embeds", img_query[i]);
        img_arguments.AddExtraParam("ef", ef);
        // img_arguments.SetExpression("ingredient_count <= " + std::to_string(ingre_num[i]) + " || instruction_step <= " + std::to_string(instruct_step[i]));
        img_arguments.SetExpression("price <= " + std::to_string(price));
        img_arguments.SetMetricType(milvus::MetricType::IP);
        milvus::SearchResults img_search_results{};
        auto status = client->Search(img_arguments, img_search_results); 
        CheckStatus("Failed to search:", status);

        for (auto& img_result : img_search_results.Results()) {
            auto& img_ids = img_result.Ids().IntIDArray();
            auto& img_distances = img_result.Scores();
            if (img_ids.size() != img_distances.size()) {
                std::cout << "img illegal result!" << std::endl;
                continue;
            }
            int len = img_ids.size();

            int rank = 0;
            for (int j = 0; j < len; j++)
                out1 << img_query_id[i] << "\t" << img_ids[j] << "\t" << ++rank << "\t" << img_distances[j] << std::endl;
            for (int j = len; j < topk; j++)
                out1 << img_query_id[i] << "\t" << -1 << "\t" << ++rank << "\t" << -1 << std::endl;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        out2 << img_query_id[i] << "\t" << std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0 << std::endl;

        int number = i + 1;
        if (number % 100 == 0)
            std::cout << number << " queries searched." << std::endl;
    }
}

int
main(int argc, char* argv[]) {
    std::vector<int> prices{3000, 9000, 300};
    std::vector<int> efs{100, 1000, 10000};

    for (int price : prices){
        for (int ef : efs){
            search(ef, price);
        }
    }
    return 0;
}
