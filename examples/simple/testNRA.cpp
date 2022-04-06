#include <iostream>
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>


bool
NRA(std::vector<std::vector<float> > ids, std::vector<std::vector<float> > scores, std::map<float, float>& RESULT, int topk) {
	float minScore = -1.0;
	bool flag = false;
	std::unordered_map<float, float> UB, LB;
	std::vector<std::unordered_map<float, float> > matrix;
	std::unordered_set<float> seenIDSet;
	matrix.resize(ids.size());
	for (int i = 0; i < ids[0].size(); i++) {
		for (int j = 0; j < ids.size(); j++) {
			matrix[j][ids[j][i]] = scores[j][i];
			seenIDSet.insert(ids[j][i]);
		}

		for (auto id : seenIDSet) {
			LB[id] = 0;
			UB[id] = 0;
			for (int k = 0; k < ids.size(); k++) {
				if (matrix[k].find(id) == matrix[k].end()) {
					LB[id] += minScore;
					UB[id] += scores[k][i];
				}
				else {
					LB[id] += matrix[k][id];
					UB[id] += matrix[k][id];
				}
			}
		}
		if (LB.size() > topk) {
			std::vector<float> LB_value;
			for (auto x : LB) LB_value.push_back(x.second);
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
		RESULT[-lb_value] = x.first;
		while (RESULT.size() > topk) {
			auto tmp = RESULT.end();
			tmp--;
			RESULT.erase(tmp);
		}
	}
	return flag;
}


int main()
{
	std::map<float, float> RESULT;
	std::vector<float> ids_0 = { 1, 2, 3, 4, 5 };
	std::vector<float> ids_1 = { 2, 3, 1, 4, 5 };
	std::vector<float> ids_2 = { 4, 3, 1, 5, 2, 9 };
	std::vector<float> scores_0 = { 1, 0.8, 0.5, 0.3, 0.1 };
	std::vector<float> scores_1 = { 0.8, 0.7, 0.3, 0.2, 0.1 };
	std::vector<float> scores_2 = { 0.8, 0.6, 0.2, 0.1, 0.0, -1.0 };

	std::vector<std::vector<float> > ids = { ids_0, ids_1, ids_2 };
	std::vector<std::vector<float> > scores = { scores_0, scores_1, scores_2 };


	bool flag = NRA(ids, scores, RESULT, 2);
	std::cout << "flag: " << flag << std::endl;

	int rank = 0;
	for (auto x : RESULT) {
		std::cout << " " << x.second << " " << ++rank << " " << -x.first << std::endl;
	}

	flag = NRA(ids, scores, RESULT, 10);
	std::cout << "flag: " << flag << std::endl;

	rank = 0;
	for (auto x : RESULT) {
		std::cout << " " << x.second << " " << ++rank << " " << -x.first << std::endl;
	}
}
