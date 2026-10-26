cmake_minimum_required(VERSION 2.8)

project(DistributedHNSW)

# MPI-only build: no Kafka (cppkafka/RdKafka), Zookeeper, or HDFS required.
option(BUILD_MPI_ONLY "Build MPI benchmarks only (bug, single_machine, ...)" ON)

add_subdirectory(${DistributedHNSW_SOURCE_DIR}/src/include/kahiplib/)

if(NOT BUILD_MPI_ONLY)
	add_subdirectory(${DistributedHNSW_SOURCE_DIR}/src/include/cppkafka/)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# commprof: no forced -std, so the binaries match the production ones
# (GNU C++17 15.3.0 -g -O2 -fopenmp, see DW_AT_producer of build/bug_*_k*).
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

find_package(PkgConfig)
pkg_check_modules(EIGEN3 eigen3)
if(EIGEN3_FOUND)
	include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

find_package(Eigen3)
if(Eigen3_FOUND)
	include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(Boost COMPONENTS system filesystem regex thread program_options date_time chrono)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenMP)
include_directories(${OpenMP_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

if(BUILD_MPI_ONLY)
	add_definitions(-DDHNSW_MPI_ONLY=1)
	message(STATUS "BUILD_MPI_ONLY=ON: skipping Kafka, Zookeeper, and HDFS")
else()
	find_package(Zookeeper REQUIRED)
	include_directories(${Zookeeper_INCLUDE_DIRS})

	set(HDFS_INCLUDE_DIR "/data/opt/hadoop-2.6.0/include/")
	set(HDFS_LIBRARY_DIR "/data/opt/hadoop-2.6.0/lib/native/")
	add_definitions(-DDHNSW_ENABLE_HDFS=1)

	INCLUDE_DIRECTORIES(${DistributedHNSW_SOURCE_DIR}/src/include ${RDKAFKA_INCLUDE_DIR} ${HDFS_INCLUDE_DIR})
	link_libraries(${HDFS_LIBRARY_DIR}/libhdfs.so)
endif()

link_libraries(-lpthread)

if(BUILD_MPI_ONLY)
	INCLUDE_DIRECTORIES(${DistributedHNSW_SOURCE_DIR}/src/include)
else()
	INCLUDE_DIRECTORIES(${DistributedHNSW_SOURCE_DIR}/src/include ${RDKAFKA_INCLUDE_DIR} ${HDFS_INCLUDE_DIR})
endif()

find_package(Faiss QUIET)

if(BUILD_MPI_ONLY)
	SET(EXAMPLES
		bug
		ground_truth
		single_machine
		recover_single_machine
		fvecs_hnsw_machine
	)
else()
	SET(EXAMPLES
		#bug2
		#start_customer
		#start_benchmark_evaluator
		#start_throughput_evaluator
		#start_customer_endless
		ground_truth
		recover_single_machine
		naive_machine
		ip_partition
		start_customer_rotate
		start_delay_evaluator
	)
endif()

FOREACH(EXAMPLE ${EXAMPLES})
	ADD_EXECUTABLE(${EXAMPLE} src/apps/${EXAMPLE}.cpp)
	if(BUILD_MPI_ONLY)
		target_link_libraries(${EXAMPLE} ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})
	else()
		target_link_libraries(${EXAMPLE} ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} ${Zookeeper_LIBRARIES} libkaffpa ${RDKAFKA_LIBRARY} cppkafka libmapping ${OpenMP_CXX_LIBRARIES})
	endif()
ENDFOREACH(EXAMPLE)

if(BUILD_MPI_ONLY AND FAISS_FOUND)
	ADD_EXECUTABLE(single_machine_mips src/apps/single_machine_mips.cpp)
	target_include_directories(single_machine_mips PRIVATE ${FAISS_INCLUDE_DIR})
	target_link_libraries(single_machine_mips ${Boost_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES} ${FAISS_LIBRARY})
	message(STATUS "Building single_machine_mips with Faiss: ${FAISS_LIBRARY}")
elseif(BUILD_MPI_ONLY)
	message(STATUS "Faiss not found; skipping single_machine_mips (run script/build_faiss_gcc10.sh)")
endif()

if(BUILD_MPI_ONLY)
	set(BUG_MIPS_DATA_DIMENSION 200 CACHE STRING "Vector dimension for bug_mips")
	set(BUG_MIPS_SIZEWORKER 2 CACHE STRING "Number of slave shards (SIZEWORKER) for bug_mips")
	set(BUG_MIPS_QUERY_SIZE 100 CACHE STRING "Number of queries (QUERY_SIZE) for bug_mips")
	set(BUG_MIPS_TOPK 100 CACHE STRING "TOPK compile-time for bug_mips")
	ADD_EXECUTABLE(bug_mips src/apps/bug_mips.cpp)
	target_compile_definitions(bug_mips PRIVATE
		DHNSW_USE_INNER_PRODUCT=1
		DATA_DIMENSION=${BUG_MIPS_DATA_DIMENSION}
		SIZEWORKER=${BUG_MIPS_SIZEWORKER}
		QUERY_SIZE=${BUG_MIPS_QUERY_SIZE}
		TOPK=${BUG_MIPS_TOPK}
	)
	target_link_libraries(bug_mips ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})
	message(STATUS "Building bug_mips: dim=${BUG_MIPS_DATA_DIMENSION} shards=${BUG_MIPS_SIZEWORKER}")
endif()

# parallel_timer (from commprof): one binary per layout, same names as the production build
# (bug_<shards>_k<topk> for DEEP/L2 dim 96, bug_mips_<shards>_k<topk> for TTI/MIPS dim 200),
# plus bug_toy for the 2-shard toy smoke test.
if(BUILD_MPI_ONLY)
	foreach(W 4 6 8)
		foreach(K 10 100)
			add_executable(bug_${W}_k${K} src/apps/bug.cpp)
			target_compile_definitions(bug_${W}_k${K} PRIVATE
				DATA_DIMENSION=96 SIZEWORKER=${W} QUERY_SIZE=10000 TOPK=${K})
			target_link_libraries(bug_${W}_k${K} ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})

			add_executable(bug_mips_${W}_k${K} src/apps/bug_mips.cpp)
			target_compile_definitions(bug_mips_${W}_k${K} PRIVATE
				DHNSW_USE_INNER_PRODUCT=1
				DATA_DIMENSION=200 SIZEWORKER=${W} QUERY_SIZE=10000 TOPK=${K})
			target_link_libraries(bug_mips_${W}_k${K} ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})
		endforeach()

		# MSMARCO Web Search: MIPS, dim 768, 9376 queries, recall@10 only
		add_executable(bug_msm_${W}_k10 src/apps/bug_mips.cpp)
		target_compile_definitions(bug_msm_${W}_k10 PRIVATE
			DHNSW_USE_INNER_PRODUCT=1
			DATA_DIMENSION=768 SIZEWORKER=${W} QUERY_SIZE=9376 TOPK=10)
		target_link_libraries(bug_msm_${W}_k10 ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})
	endforeach()

	add_executable(bug_toy src/apps/bug.cpp)
	target_compile_definitions(bug_toy PRIVATE
		DATA_DIMENSION=32 SIZEWORKER=2 QUERY_SIZE=10 TOPK=10)
	target_link_libraries(bug_toy ${Boost_LIBRARIES} ${MPI_C_LIBRARIES} libkaffpa libmapping ${OpenMP_CXX_LIBRARIES})
endif()
