# Copyright (c) Zili Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

DIST_SRC=$(wildcard *.cpp)
DIST=$(filter-out shard_bundle.cpp shard_calibrate.cpp,$(DIST_SRC:.cpp=))

MULTI_OBJS=shard_bundle.cpp

all: $(DIST)

clean:
	rm -f $(DIST)

build_shard: build_shard.cpp shard_bundle.cpp shard_calibrate.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ ../libfaiss.a $(LDFLAGS) $(LIBS)

worker: worker.cpp shard_bundle.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ ../libfaiss.a $(LDFLAGS) $(LIBS)

master: master.cpp bound_calibrate.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ ../libfaiss.a $(LDFLAGS) $(LIBS)

%: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ ../libfaiss.a $(LDFLAGS) $(LIBS)

.PHONY: all clean
