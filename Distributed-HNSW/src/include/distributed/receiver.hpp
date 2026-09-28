#pragma once

#include <mpi.h>
#include <chrono>
#include <unordered_map>
#include <queue>
#include "distributed/slave.hpp"
#include "mpi_executor.hpp"
#include "parameters.hpp"

using std::unordered_map;
using std::priority_queue;
using std::pair;
using std::vector;
using std::make_pair;

namespace mt {

    void receiveResultMessage(void* buffer, int msg_len);

    struct Answer {
        Answer(): n_slaves(0) {};
        int n_slaves;
        priority_queue<pair<float, int>> p_queue;
    };

    class Receiver {
        int _n_queries;        // for benchmarking & debug
        unordered_map<int, Answer> _query_map;
        vector<vector<pair<float, int>>> _result;

        void _insert_answers(priority_queue<pair<float, int>>& pq, int* index, float* dist) {
            /* Inserts candidate vectors into priority queue. An vector will be
                ignored if it is not in the top n results. */
            for (int i=0; i< TOPK; i++) {
                // Check if we should insert the vector
                if (pq.size() < TOPK || dist[i] < pq.top().first) {
                    if (pq.size() >= TOPK) pq.pop();
                    pq.push(make_pair(dist[i], index[i]));
                }
            }
        }

        void _commit_answers(priority_queue<pair<float, int>>& pq, int query_id) {
            /* Gets all elements from the priority queue and commit them to result stack */
            vector<pair<float, int>> indexes;
            while(!pq.empty()) {
                indexes.push_back(pq.top());
                pq.pop();
            }
            _result[query_id] = indexes;
        }

    public:
        explicit Receiver(int n_queries): _n_queries(n_queries) {
            _result.resize(n_queries);
        };

        // Times only the query phase on the receiver (steady_clock, single node):
        // from entering the receive loop until the last query is committed.
        // Excludes data/index load. Does NOT use cross-node MPI_Wtime().
        // avg_time = query_phase_wall_s / n_queries (throughput-style mean).
        vector<vector<pair<float, int>>> receive(double& avg_time) {
            int ef = -1;
            int counter = 0;
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            while (counter < _n_queries) {
                int msg_len = sizeof(mt::result_message);
                void* buffer = malloc(msg_len);
                mt::receiveResultMessage(buffer, msg_len);
                mt::result_message* result_msg = (mt::result_message*)buffer;
                Answer& answer = _query_map[result_msg->query_id];
                ef = result_msg->ef;
                _insert_answers(answer.p_queue, result_msg->result_id, result_msg->dist);
                answer.n_slaves ++;
                if (answer.n_slaves == result_msg->num_wake_up) {
                    _commit_answers(answer.p_queue, result_msg->query_id);
                    _query_map.erase(result_msg->query_id);
                    counter++;
                }
                free(buffer);
            }
            const double query_phase_s =
                std::chrono::duration<double>(clock::now() - t0).count();
            avg_time = query_phase_s / static_cast<double>(_n_queries);
            return _result;
        }

    };

}
