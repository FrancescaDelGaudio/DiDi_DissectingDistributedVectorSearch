/*
 * slave.hpp
 *
 *  Created on: 22 Mar 2019
 *      Author: oruqimaru
 */

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "distributed/hnsw_metric.hpp"
#include "distributed/macro.h"
#include "distributed/sender.hpp"
#include "hnswlib/hnswalg.h"

using std::cout;
using std::endl;
using std::pair;
using std::priority_queue;
using std::string;
using std::vector;

namespace mt {

	static const int SLAVE_HNSW_M = 32;
	static const int SLAVE_HNSW_EF_CONSTRUCTION = 100;

	typedef struct rm{
		int result_id[TOPK];
		float dist[TOPK];
		int query_id;
		double start_time;
		int num_wake_up;
		int ef;
	} result_message;

	class Slave{
	private:
		int _subset_size;
		vector<int> _subset;
		vector<float> _vectors;
		dhnsw::HnswSpace _space;
		std::unique_ptr<hnswlib::HierarchicalNSW<float>> _hnsw;

		static void loadFvecsSubsetVectors(
			const string &base_path,
			int dim,
			const vector<int> &ids,
			vector<float> &out) {
			const size_t n = ids.size();
			out.resize(n * static_cast<size_t>(dim));
			const size_t record_size =
				sizeof(int) + static_cast<size_t>(dim) * sizeof(float);
			std::ifstream fin(base_path, std::ios::binary);
			if (!fin) {
				cout << "#[slav] cannot open base data: " << base_path << endl;
				std::exit(1);
			}
			for (size_t i = 0; i < n; ++i) {
				fin.seekg(static_cast<std::streamoff>(ids[i]) *
				          static_cast<std::streamoff>(record_size));
				int file_dim = 0;
				fin.read(reinterpret_cast<char *>(&file_dim), sizeof(int));
				if (!fin || file_dim != dim) {
					cout << "#[slav] bad fvecs record at id " << ids[i] << endl;
					std::exit(1);
				}
				float *dst = &out[i * static_cast<size_t>(dim)];
				fin.read(reinterpret_cast<char *>(dst),
				         static_cast<std::streamsize>(dim) * sizeof(float));
				if (!fin) {
					cout << "#[slav] short read at id " << ids[i] << endl;
					std::exit(1);
				}
			}
		}

		Slave(const string &path, const string &subset_path, int ef, bool build_from_fvecs):
				_space(DATA_DIMENSION),
				_hnsw(nullptr){
			if (!build_from_fvecs) {
				loadSubset(subset_path);
				_hnsw.reset(new hnswlib::HierarchicalNSW<float>(&_space, path));
				_hnsw->setEf(ef);
				return;
			}
			loadSubset(subset_path);
			loadFvecsSubsetVectors(path, DATA_DIMENSION, _subset, _vectors);
			_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
				&_space, _subset_size, SLAVE_HNSW_M, SLAVE_HNSW_EF_CONSTRUCTION));
			cout << "#[slav] inside constructor, before adding points in hnsw ("
			     << _subset_size << " points, efConstruction="
			     << SLAVE_HNSW_EF_CONSTRUCTION << ")" << endl;
			if (_subset_size > 0) {
				_hnsw->addPoint(
					(void *)&_vectors[0],
					static_cast<size_t>(_subset[0]));
			}
#pragma omp parallel for schedule(static)
			for (int i = 1; i < _subset_size; ++i) {
				_hnsw->addPoint(
					(void *)&_vectors[static_cast<size_t>(i) * DATA_DIMENSION],
					static_cast<size_t>(_subset[i]));
			}
			cout << "#[slav] points added in hnsw" << endl;
			_hnsw->setEf(ef);
		}

	public:
		static Slave *loadFromIndex(const string &hnsw_path, const string &subset_path, int ef) {
			return new Slave(hnsw_path, subset_path, ef, false);
		}

		static Slave *buildFromSubset(const string &base_path, const string &subset_path, int ef) {
			return new Slave(base_path, subset_path, ef, true);
		}

		void makeResult(task_message& task, result_message& result){
			_hnsw->setEf(task.ef);
			priority_queue<pair<float, long unsigned int >> topk =
				_hnsw->searchKnn(task.data, TOPK);
			for (int i = 0; i < TOPK; i++){
				result.result_id[i] = (int) topk.top().second;
				result.dist[i] = topk.top().first;
				topk.pop();
			}
			result.ef = task.ef;
			result.num_wake_up = task.num_wake_up;
			result.query_id = task.index;
			result.start_time = task.start_time;
		}

		// parallel_timer: called by the main thread before a parallel batch.
		void setEf(int ef){
			_hnsw->setEf(ef);
		}

		// parallel_timer: same as makeResult without setEf, safe to call from many
		// threads at once (hnswlib searchKnn is const; the visited-list pool is locked).
		void search(const task_message& task, result_message& result) const {
			priority_queue<pair<float, long unsigned int >> topk =
				_hnsw->searchKnn(task.data, TOPK);
			for (int i = 0; i < TOPK; i++){
				result.result_id[i] = (int) topk.top().second;
				result.dist[i] = topk.top().first;
				topk.pop();
			}
			result.ef = task.ef;
			result.num_wake_up = task.num_wake_up;
			result.query_id = task.index;
			result.start_time = task.start_time;
		}

		void saveHNSW(string path){
			_hnsw->saveIndex(path);
		}

		int loadSubset(string subset_path){
			std::ifstream inFile(subset_path);
			inFile >> _subset_size;
			_subset.resize(_subset_size);
			for (int i = 0; i < _subset_size; i++){
				inFile >> _subset[i];
			}
			inFile.close();
			return _subset_size;
		}
	};
}
