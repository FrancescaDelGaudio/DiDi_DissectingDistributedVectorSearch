// Copyright (c) Zili Zhang.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/time.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bound_calibrate.h"

#define MAXLINE 4096

namespace {

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

bool parse_host_port(const std::string& endpoint, std::string* host, int* port) {
  size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos) {
    *host = endpoint;
    *port = 3456;
    return true;
  }
  *host = endpoint.substr(0, colon);
  *port = std::stoi(endpoint.substr(colon + 1));
  return true;
}

void usage(const char* prog) {
  fprintf(stderr,
      "Usage:\n"
      "  %s train <host:port> [host:port ...]\n"
      "  %s search <output_dir> [options] <host:port> [host:port ...]\n"
      "\n"
      "Search options (§4.5 error-bound calibration on leader):\n"
      "  --global-recall R   Global recall target (default 0.9, i.e. ε=10%%)\n"
      "  --k K               Top-k for calibration (required for calibrated search)\n"
      "  --gamma G           Confidence γ (default 0.999)\n"
      "  --no-calibrate      Send plain 'search' (workers use their local recall)\n"
      "\n"
      "Examples:\n"
      "  %s train 127.0.0.1:3456\n"
      "  %s search ./results --global-recall 0.6 --k 5 127.0.0.1:3456 127.0.0.1:3457\n",
      prog, prog, prog, prog);
}

bool recv_until_marker(int sockfd,
                       std::string* carry,
                       const std::string& marker,
                       FILE* fp) {
  while (true) {
    size_t pos = carry->find(marker);
    if (pos != std::string::npos) {
      if (fp && pos > 0)
        fwrite(carry->data(), 1, pos, fp);
      carry->erase(0, pos + marker.size());
      return true;
    }

    if (carry->size() > marker.size()) {
      size_t flush = carry->size() - (marker.size() - 1);
      if (fp)
        fwrite(carry->data(), 1, flush, fp);
      carry->erase(0, flush);
    }

    char buffer[MAXLINE];
    int n = recv(sockfd, buffer, MAXLINE, 0);
    if (n <= 0)
      return false;
    carry->append(buffer, n);
  }
}

bool call_worker(const std::string& host,
                 int port,
                 const std::string& ins,
                 int worker_index,
                 const std::string& output_dir) {
  int sockfd;
  char recvline[4096];
  struct sockaddr_in servaddr;

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("create socket error: %s(errno: %d)\n", strerror(errno), errno);
    return false;
  }

  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &servaddr.sin_addr) <= 0) {
    printf("inet_pton error for %s\n", host.c_str());
    close(sockfd);
    return false;
  }

  if (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
    printf("connect error to %s:%d: %s(errno: %d)\n",
           host.c_str(), port, strerror(errno), errno);
    close(sockfd);
    return false;
  }

  printf("send '%s' to %s:%d\n", ins.c_str(), host.c_str(), port);
  if (send(sockfd, ins.c_str(), ins.size(), 0) < 0) {
    printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
    close(sockfd);
    return false;
  }

  const bool is_search = ins.compare(0, 6, "search") == 0;
  if (is_search) {
    std::string file_end = "file done";
    std::string file_idx = std::to_string(worker_index) + "idx.ivecs";
    std::string file_dis = std::to_string(worker_index) + "dis.fvecs";
    if (!output_dir.empty() && output_dir.back() != '/')
      file_idx = output_dir + "/" + file_idx;
    else if (!output_dir.empty())
      file_idx = output_dir + file_idx;
    else
      file_idx = "./" + file_idx;

    if (!output_dir.empty() && output_dir.back() != '/')
      file_dis = output_dir + "/" + file_dis;
    else if (!output_dir.empty())
      file_dis = output_dir + file_dis;
    else
      file_dis = "./" + file_dis;

    std::string carry;
    FILE* fp = fopen(file_idx.c_str(), "wb");
    if (!fp || !recv_until_marker(sockfd, &carry, file_end, fp)) {
      if (fp) fclose(fp);
      close(sockfd);
      return false;
    }
    fclose(fp);

    FILE* fp2 = fopen(file_dis.c_str(), "wb");
    if (!fp2 || !recv_until_marker(sockfd, &carry, file_end, fp2)) {
      if (fp2) fclose(fp2);
      close(sockfd);
      return false;
    }
    fclose(fp2);
    printf("Saved %s and %s\n", file_idx.c_str(), file_dis.c_str());

    // Remaining bytes in carry should be the final status message.
    if (carry.empty()) {
      char buffer[MAXLINE];
      int n = recv(sockfd, buffer, MAXLINE, 0);
      if (n > 0)
        carry.append(buffer, n);
    }
    printf("recv from %s:%d: %s\n", host.c_str(), port, carry.c_str());
    bool ok = carry.find("success") != std::string::npos;
    close(sockfd);
    return ok;
  }

  int tmp = recv(sockfd, recvline, MAXLINE, 0);
  if (tmp <= 0) {
    close(sockfd);
    return false;
  }
  recvline[tmp] = '\0';
  printf("recv from %s:%d: %s\n", host.c_str(), port, recvline);
  bool ok = std::string(recvline) == "success";
  close(sockfd);
  return ok;
}

}  // namespace

struct SearchOptions {
  double global_recall = 0.9;
  int k = 0;
  double gamma = 0.999;
  bool calibrate = true;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  std::string mode = argv[1];
  std::vector<std::string> endpoints;
  std::string output_dir;
  SearchOptions search_opts;

  if (mode == "train") {
    for (int i = 2; i < argc; i++) endpoints.push_back(argv[i]);
  } else if (mode == "search") {
    if (argc < 4) {
      usage(argv[0]);
      return 1;
    }
    output_dir = argv[2];
    for (int i = 3; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--global-recall") {
        if (i + 1 >= argc) {
          usage(argv[0]);
          return 1;
        }
        search_opts.global_recall = std::stod(argv[++i]);
      } else if (arg == "--k") {
        if (i + 1 >= argc) {
          usage(argv[0]);
          return 1;
        }
        search_opts.k = std::stoi(argv[++i]);
      } else if (arg == "--gamma") {
        if (i + 1 >= argc) {
          usage(argv[0]);
          return 1;
        }
        search_opts.gamma = std::stod(argv[++i]);
      } else if (arg == "--no-calibrate") {
        search_opts.calibrate = false;
      } else {
        endpoints.push_back(arg);
      }
    }
    if (endpoints.empty()) {
      fprintf(stderr, "search requires at least one worker host:port\n");
      usage(argv[0]);
      return 1;
    }
    if (search_opts.calibrate && search_opts.k <= 0) {
      fprintf(stderr,
              "error: --k is required for §4.5 calibration (or pass --no-calibrate)\n");
      return 1;
    }
  } else {
    usage(argv[0]);
    return 1;
  }

  std::string search_cmd = mode;
  if (mode == "search" && search_opts.calibrate) {
    const int w = static_cast<int>(endpoints.size());
    const double global_epsilon = 1.0 - search_opts.global_recall;
    const double epsilon_c = auncel_dist::calibrate_local_epsilon(
        global_epsilon, search_opts.k, w, search_opts.gamma);
    const double local_recall = 1.0 - epsilon_c;
    const double prob = auncel_dist::probability_meets_global_bound(
        global_epsilon, search_opts.k, w, epsilon_c);

    char buf[128];
    snprintf(buf, sizeof(buf), "search %.9g", local_recall);
    search_cmd = buf;

    printf("§4.5 calibration: w=%d k=%d global_recall=%.6f (ε=%.6f)\n",
           w, search_opts.k, search_opts.global_recall, global_epsilon);
    printf("  calibrated local εc=%.6f -> local_recall=%.6f\n",
           epsilon_c, local_recall);
    printf("  P(meet global ε)=%.6f (γ=%.6f)\n", prob, search_opts.gamma);
  }

  double t0 = elapsed();
  bool all_ok = true;
  for (size_t i = 0; i < endpoints.size(); i++) {
    std::string host;
    int port = 3456;
    parse_host_port(endpoints[i], &host, &port);
    const std::string& cmd = (mode == "search") ? search_cmd : mode;
    if (!call_worker(host, port, cmd, static_cast<int>(i), output_dir))
      all_ok = false;
  }

  printf("%s finished in %.3f s (%s)\n",
         mode.c_str(), elapsed() - t0, all_ok ? "ok" : "errors");
  return all_ok ? 0 : 1;
}
