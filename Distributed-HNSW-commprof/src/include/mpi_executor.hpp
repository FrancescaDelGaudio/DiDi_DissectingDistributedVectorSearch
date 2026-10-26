/*
 * mpi_executor.hpp
 *
 *  Created on: 22 Mar 2019
 *      Author: oruqimaru
 */

#pragma once

#include <algorithm>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/errors.hpp>

#include "dhnswlib/time.hpp"
#include "bench/bencher.hpp"
#include "bench/prober.hpp"
#include "parameters.hpp"
#include "matrix.hpp"
#include "metric.hpp"
#include "utils/util.hpp"
#include "utils/cluster.hpp"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "waker/waker.hpp"
#include "distributed/sender.hpp"
#include "distributed/slave.hpp"
#include "distributed/receiver.hpp"
#include "distributed/macro.h"

using std::ifstream;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/*
 * status code with two digit:
 * first for sender: 0 load hnsw 1 retrain hnsw and save
 * second for slave: 0 load hnsw 1 retrain hnsw and save
 */

namespace mt {

	class MpiPartition{
	private:
		int _dim;
		int _total_partition;
		vector<int> _map;
		hnswlib::L2Space _l2space;
		hnswlib::HierarchicalNSW<float> _hnsw;

		int load_map(string map_path){
			ifstream fin(map_path);
			fin >> _total_partition;
			int iter, buffer;
			fin >> iter;
			for (int i = 0; i < iter; i++){
				fin >> buffer;
				_map.push_back(buffer);
			}
			return iter;
		}
	public:
		MpiPartition(int dim, string hnsw_path, string map_path):_dim(dim), _l2space(dim),_hnsw(&_l2space, hnsw_path){
			int size = load_map(map_path);
			assert(size == _hnsw.max_elements_);
		}

		int searchHnsw(float* query){
			std::priority_queue<std::pair<float, long unsigned int > > result = _hnsw.searchKnn(query, 1);
			return _map[result.top().second];
		}

		int getTotalPartition(){
			return _total_partition;
		}
	};

	/*
	 * commprof: communication / search time breakdown.
	 * All timers are per-process steady_clock durations (never compared across nodes).
	 * Output lines start with "#[prof ]".
	 */
	static const int PROF_PING_INDEX = -2;   // task_message.index for an empty ping (no search)
	static const int PROF_PING_TAG = 2;      // slave -> sender ping reply
	static const int PROF_PING_ITERS = 1000; // timed pings per slave

	typedef std::chrono::steady_clock prof_clock;

	inline double profSeconds(prof_clock::time_point a, prof_clock::time_point b){
		return std::chrono::duration<double>(b - a).count();
	}

	inline string profHostname(){
		char buf[256];
		if (gethostname(buf, sizeof(buf)) != 0)
			return "?";
		buf[sizeof(buf) - 1] = 0;
		return string(buf);
	}

	inline double profPercentile(vector<double> v, double p){
		if (v.empty())
			return 0.0;
		std::sort(v.begin(), v.end());
		size_t idx = (size_t)(p * (double)(v.size() - 1));
		return v[idx];
	}

	void sendMessage(int index, Sender* sender, int& counter, int sender_ef, int wake_up_controller, double& route_s, double& send_s){
		task_message tm;
		vector<int> destinations;
		const prof_clock::time_point t0 = prof_clock::now();
		sender->makeTask(index, destinations, tm, MPI_Wtime(), sender_ef, wake_up_controller);
		const prof_clock::time_point t1 = prof_clock::now();
		counter += tm.num_wake_up;
		for (int j = 0; j < destinations.size(); j++)
			MPI_Send((void*) &tm, sizeof(task_message), MPI_BYTE, destinations[j], TASK_TAG, MPI_COMM_WORLD);
		const prof_clock::time_point t2 = prof_clock::now();
		route_s += profSeconds(t0, t1);
		send_s += profSeconds(t1, t2);
	}

	/*
	 * Empty round trip sender -> slave -> sender with the real message sizes and no search:
	 * the practical MPI floor for one task + one result on this fabric.
	 * Tasks are FIFO per (sender, slave) pair, so the untimed warm-up pings only return
	 * once the slave has drained every real query.
	 */
	void profPingSlaves(){
		task_message ping;
		memset(&ping, 0, sizeof(ping));
		ping.index = PROF_PING_INDEX;
		result_message pong;
		for (int i = 0; i < SIZEWORKER; i++){
			MPI_Send((void*) &ping, sizeof(task_message), MPI_BYTE, i, TASK_TAG, MPI_COMM_WORLD);
			MPI_Recv((void*) &pong, sizeof(result_message), MPI_BYTE, i, PROF_PING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		for (int i = 0; i < SIZEWORKER; i++){
			vector<double> rtt_us(PROF_PING_ITERS);
			double sum = 0;
			for (int it = 0; it < PROF_PING_ITERS; it++){
				const prof_clock::time_point t0 = prof_clock::now();
				MPI_Send((void*) &ping, sizeof(task_message), MPI_BYTE, i, TASK_TAG, MPI_COMM_WORLD);
				MPI_Recv((void*) &pong, sizeof(result_message), MPI_BYTE, i, PROF_PING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				rtt_us[it] = profSeconds(t0, prof_clock::now()) * 1e6;
				sum += rtt_us[it];
			}
			char line[512];
			snprintf(line, sizeof(line),
				"#[prof ] ping slave%d n=%d task_B=%zu result_B=%zu rtt_us_mean=%.2f rtt_us_p50=%.2f rtt_us_p99=%.2f rtt_us_min=%.2f rtt_us_max=%.2f\n",
				i, PROF_PING_ITERS, sizeof(task_message), sizeof(result_message),
				sum / PROF_PING_ITERS, profPercentile(rtt_us, 0.50), profPercentile(rtt_us, 0.99),
				profPercentile(rtt_us, 0.0), profPercentile(rtt_us, 1.0));
			cout << line << std::flush;
		}
	}

	void sendShutdownToSlaves(){
		task_message shutdown;
		shutdown.index = -1;
		for (int i = 0; i < SIZEWORKER; i++)
			MPI_Send((void*) &shutdown, sizeof(task_message), MPI_BYTE, i, TASK_TAG, MPI_COMM_WORLD);
	}

	void printQueryResults(const Bencher& truth, const Bencher& got){
		int n = std::min(truth.size(), got.size());
		for (int i = 0; i < n; i++){
			const BenchRecord& truth_rec = truth.getRecord(i);
			const BenchRecord& got_rec = got.getRecord(i);
			cout << "#[query " << i << "] recall=" << truth_rec.recall(got_rec) << endl;
			cout << "  returned:";
			for (const auto& p : got_rec.getKNN())
				cout << " id=" << p.second << " dist=" << p.first;
			cout << endl;
			cout << "  truth:   ";
			for (const auto& p : truth_rec.getKNN())
				cout << " id=" << p.second << " dist=" << p.first;
			cout << endl;
		}
	}

	void receiveResultMessage(void* buffer, int msg_len){
		MPI_Recv(buffer, msg_len, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

#ifdef DHNSW_ENABLE_HDFS
	void mpiPartitioner(ss::parameter& para){
		/*
		 * using para
		 * base_data out_dir containing /hnsw/partition /partition_map
		 */
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//		if (world_size != SIZEWORKER){
//			cout << "#[error ] wrong number process initialled with" + std::to_string(world_size) << endl;
//			MPI_Abort(MPI_COMM_WORLD, 0);
//		}

		long long start_time = dhnsw::get_current_time_milliseconds();
		ss::Matrix<float> data(para.hdfs_host, para.hdfs_port, para.base_data, world_rank, world_size, para.base_size);
		long long load_time = dhnsw::get_current_time_milliseconds();
		cout << "#[timer] load file use " + std::to_string(load_time - start_time) + " milisecond\n";

		MpiPartition partitioner(data.getDim(), para.out_dir + "/hnsw/partition", para.out_dir + "/partition_map");
		long long construct_time = dhnsw::get_current_time_milliseconds();
		cout << "#[timer] construct partitioner use " + std::to_string(construct_time - load_time) + " milisecond\n";

		omp_set_num_threads(20);
		int sizer = data.getSize();
		vector<int> result(sizer);
		cout << "start partition\n";
#pragma omp parallel for
		for (int i = 0; i < sizer; i++){
			result[i] = partitioner.searchHnsw(data[i]);
		}

		long long partition_time = dhnsw::get_current_time_milliseconds();
		cout << "#[timer] partition use " + std::to_string(partition_time - construct_time) + " milisecond\n";
		cout << "#[worker]"+std::to_string(world_rank) + " start to save\n";

		vector<std::ofstream* > fouts;
		for (int i = 0; i < partitioner.getTotalPartition(); i++){
			std::ofstream* fout = new std::ofstream(para.out_dir + "/subfile/w" + std::to_string(world_rank) + "/partition" + std::to_string(i), std::iostream::binary);
			fouts.push_back(fout);
		}
		int dimer = data.getDim();
		for (int i = 0; i < sizer; i++){
			std::ofstream* fout = fouts[result[i]];
			fout->write((char*)&dimer, sizeof(int));
			fout->write((char*)data[i], dimer * sizeof(float));
			fout->write((char*)&data.id_[i], sizeof(int));
		}
		for(int i = 0; i < partitioner.getTotalPartition(); i++){
			fouts[i]->close();
			delete fouts[i];
		}
		long long save_time = dhnsw::get_current_time_milliseconds();
		cout << "#[timer] save use " + std::to_string(save_time - partition_time) + " milisecond\n";
	}
#endif



    void mpiBody(ss::parameter& para, mt::Partition& partition){
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);


		if (world_size != SIZEWORKER + 2){
			cout << "#[error ] wrong number process initialled" << endl;
			MPI_Abort(MPI_COMM_WORLD, 0);
		}

		if (world_rank == world_size - 2) {
		    void* v;
		    int flag;
		    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &v, &flag);
		    if(flag) {
		    	int vval = *(int*)v;
		    	cout << "MPI_WTIME_IS_GLOBAL is " + std::to_string(vval) << endl;
		    }
		    else {
		    	cout << "MPI_WTIME_IS_GLOBAL NOT SET" << endl;
		    }
			//logic for task sender
			cout << "#[mpi ] sender start" << endl;
			ss::Matrix<float> query(para.query_data);
			mt::Sender* sender;
			vector<vector<float> > centroids;
			loadCentroids(centroids, para.centroids_file);
			// Always load the centroid->shard map written at indexing time.
			// Re-running KaHIP at query time is nondeterministic vs the saved
			// hnsw_slave* placement and destroys recall (esp. with 6+ shards).
			vector<int> centroid_shard_map;
			{
				string map_path = para.map_address.empty() ? para.hnsw_dir + "/map" : para.map_address;
				loadCentroidShardMap(centroid_shard_map, map_path);
				cout << "#[mpi ] loaded centroid shard map from " << map_path << endl;
			}
#ifdef DHNSW_USE_INNER_PRODUCT
			if (para.mode_code / 10){
				sender = new Sender(world_size - 2, query, centroids, centroid_shard_map, DATA_DIMENSION, centroids.size(), SIZEWORKER, para.sender_ef);
				sender->saveHNSW(para.hnsw_dir + "/hnsw_sender");
			}
			else
				sender = new Sender(world_size - 2, query, para.hnsw_dir + "/hnsw_meta", centroids, centroid_shard_map, DATA_DIMENSION, centroids.size(), SIZEWORKER, para.sender_ef);
#else
			if (para.mode_code / 10){
				sender = new Sender(world_size - 2, query, centroids, centroid_shard_map, DATA_DIMENSION, centroids.size(), SIZEWORKER, para.sender_ef);
				sender->saveHNSW(para.hnsw_dir + "/hnsw_sender");
			}
			else
				sender = new Sender(world_size - 2, query, para.hnsw_dir + "/hnsw_meta", centroids, centroid_shard_map, DATA_DIMENSION, centroids.size(), SIZEWORKER, para.sender_ef);
#endif
			// Only rewrite subset files when (re)building slaves. Query-only
			// mode must keep the subsets that match the on-disk hnsw_slave*.
			if (para.mode_code % 10) {
				vector<vector<int> > clusters;
				mt::loadClusters(clusters, para.cluster_file);
				for (int i = 0; i < SIZEWORKER; i++){
					vector<int> members = sender->_waker.getMember(i);
					vector<int> subset;
					for (int j = 0; j < members.size(); j++)
						subset.insert(subset.end(), clusters[members[j]].begin(), clusters[members[j]].end());
					std::sort(subset.begin(), subset.end());
					subset.erase(std::unique(subset.begin(), subset.end()), subset.end());
					saveSubset(i, subset, para.subset_dir);
				}
			}


			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			int total_wakeup = 0;
			int wakeup_k = para.wake_up_controller > 0 ? para.wake_up_controller : 10;
			double prof_route_s = 0, prof_send_s = 0;
			const prof_clock::time_point prof_t0 = prof_clock::now();
			for (int i = 0; i < query.getSize(); i++)
				sendMessage(i, sender, total_wakeup, para.sender_ef, wakeup_k, prof_route_s, prof_send_s);
			const double prof_total_s = profSeconds(prof_t0, prof_clock::now());
			{
				char line[512];
				snprintf(line, sizeof(line),
					"#[prof ] sender host=%s queries=%d task_msgs=%d task_B=%zu total_s=%.6f route_s=%.6f send_s=%.6f route_us_per_query=%.2f send_us_per_msg=%.2f\n",
					profHostname().c_str(), query.getSize(), total_wakeup, sizeof(task_message),
					prof_total_s, prof_route_s, prof_send_s,
					prof_route_s * 1e6 / std::max(1, query.getSize()),
					prof_send_s * 1e6 / std::max(1, total_wakeup));
				cout << line << std::flush;
			}
			cout << "#[sender ] finish sending all the task with wake rate: " + std::to_string(total_wakeup * 1.0 / (SIZEWORKER * query.getSize() * 1.0)) << endl;
			{
				vector<int>* wake_counter = sender->_waker.getWakeCounter();
				cout << "#[sender ] per-slave query counts:";
				int min_c = (*wake_counter)[0], max_c = (*wake_counter)[0], sum_c = 0;
				for (int i = 0; i < SIZEWORKER; i++) {
					int c = (*wake_counter)[i];
					cout << " slave" << i << "=" << c;
					sum_c += c;
					if (c < min_c) min_c = c;
					if (c > max_c) max_c = c;
				}
				cout << endl;
				cout << "#[sender ] total_wakeup=" << total_wakeup
				     << " avg_per_slave=" << (sum_c * 1.0 / SIZEWORKER)
				     << " min=" << min_c
				     << " max=" << max_c << endl;
			}
			profPingSlaves();
			sendShutdownToSlaves();
			delete sender;
		} else if (world_rank == world_size - 1) {
			//logic for result receiver
			Bencher truth_bench(para.ground_truth.c_str());
			mt::Receiver receiver(QUERY_SIZE);
            MPI_Barrier(MPI_COMM_WORLD); // wait for sender to cluster
            MPI_Barrier(MPI_COMM_WORLD); // wait for slaves to construct HNSW
                double avg_time;
				vector<vector<pair<float, int>>> result = receiver.receive(avg_time);
				Bencher current_bench(result, false);
				const double recall = truth_bench.avg_recall(current_bench);
				const double query_phase_s = avg_time * static_cast<double>(QUERY_SIZE);
				const double qps = (query_phase_s > 0.0)
					? (static_cast<double>(QUERY_SIZE) / query_phase_s) : 0.0;
				// avg_time = query-phase wall / n_queries (receiver steady_clock; load excluded)
				cout << "time || recall"<< endl << std::to_string(avg_time) + " " + std::to_string(recall) << endl;
				cout << "#[timer] query_phase_wall_s=" << query_phase_s
				     << " avg_s_per_query=" << avg_time
				     << " qps=" << qps
				     << " n_queries=" << QUERY_SIZE << endl;
				if (para.print_results)
					printQueryResults(truth_bench, current_bench);
		} else {
			//logic for slaves
			MPI_Barrier(MPI_COMM_WORLD);
			mt::Slave* slave;
			if (para.mode_code % 10){
				slave = Slave::buildFromSubset(para.base_data, para.subset_dir + "/slave" + std::to_string(world_rank), para.slave_ef);
				slave->saveHNSW(para.hnsw_dir + "/hnsw_slave" + std::to_string(world_rank));
			}
			else
				slave = Slave::loadFromIndex(para.hnsw_dir + "/hnsw_slave" + std::to_string(world_rank), para.subset_dir + "/slave" + std::to_string(world_rank), para.slave_ef);
			MPI_Barrier(MPI_COMM_WORLD);
			// commprof: recv_ready = task was already queued locally when we asked for it
			// (pure MPI receive cost); recv_wait = slave idle, starved by the sender.
			// window = first task received .. last result sent.
			long prof_tasks = 0, prof_ready = 0, prof_waited = 0;
			double prof_search_s = 0, prof_send_s = 0, prof_recv_ready_s = 0, prof_recv_wait_s = 0;
			prof_clock::time_point prof_first, prof_last;
			while(true){
				task_message tm;
				int prof_flag = 0;
				const prof_clock::time_point t0 = prof_clock::now();
				MPI_Iprobe(world_size - 2, TASK_TAG, MPI_COMM_WORLD, &prof_flag, MPI_STATUS_IGNORE);
				MPI_Recv((void*) &tm,sizeof(task_message), MPI_BYTE, world_size - 2, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				const prof_clock::time_point t1 = prof_clock::now();
				if (tm.index == PROF_PING_INDEX){
					result_message pong;
					memset(&pong, 0, sizeof(pong));
					pong.query_id = PROF_PING_INDEX;
					MPI_Send((void*) &pong, sizeof(result_message), MPI_BYTE, world_size - 2, PROF_PING_TAG, MPI_COMM_WORLD);
					continue;
				}
				if (tm.index < 0)
					break;
				result_message rm;
				slave->makeResult(tm, rm);
				const prof_clock::time_point t2 = prof_clock::now();
				MPI_Send((void*) &rm, sizeof(result_message), MPI_BYTE, world_size - 1, RESULT_TAG, MPI_COMM_WORLD);
				const prof_clock::time_point t3 = prof_clock::now();
				if (prof_tasks == 0)
					prof_first = t1;
				else if (prof_flag){
					prof_ready++;
					prof_recv_ready_s += profSeconds(t0, t1);
				} else {
					prof_waited++;
					prof_recv_wait_s += profSeconds(t0, t1);
				}
				prof_tasks++;
				prof_search_s += profSeconds(t1, t2);
				prof_send_s += profSeconds(t2, t3);
				prof_last = t3;
			}
			{
				const double window_s = prof_tasks > 0 ? profSeconds(prof_first, prof_last) : 0.0;
				const double n = (double) std::max(1L, prof_tasks);
				char line[768];
				snprintf(line, sizeof(line),
					"#[prof ] slave%d host=%s tasks=%ld result_B=%zu window_s=%.6f search_s=%.6f send_s=%.6f recv_ready_s=%.6f recv_wait_s=%.6f n_recv_ready=%ld n_recv_wait=%ld search_us_per_task=%.2f send_us_per_task=%.2f recv_ready_us_per_task=%.2f\n",
					world_rank, profHostname().c_str(), prof_tasks, sizeof(result_message),
					window_s, prof_search_s, prof_send_s, prof_recv_ready_s, prof_recv_wait_s,
					prof_ready, prof_waited,
					prof_search_s * 1e6 / n, prof_send_s * 1e6 / n,
					prof_recv_ready_s * 1e6 / (double) std::max(1L, prof_ready));
				cout << line << std::flush;
			}
			delete slave;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
	}

}



