#include "rdma/ip_config.h"

#include <unistd.h>

// Get local machine ip addr.
std::string get_local_ip_addr() {
  struct ifaddrs *ifaddr, *ifa;
  char ip[INET_ADDRSTRLEN];

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return "";
  }

  auto usable_ipv4 = [](const std::string &ip, const std::string &iface) {
    if (iface == "lo") return false;
    // HPC nodes often expose IB/RoCE link-local (169.254.x.x) before the
    // cluster data-plane IP; CoTra matches DNS-resolved host IPs in .conf.
    if (ip.rfind("127.", 0) == 0) return false;
    if (ip.rfind("169.254.", 0) == 0) return false;
    return true;
  };

  std::string fallback;
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {  // IPv4
      struct sockaddr_in *sa = (struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &(sa->sin_addr), ip, INET_ADDRSTRLEN);
      std::string ipStr(ip);
      std::string ifaceName(ifa->ifa_name);
      if (!usable_ipv4(ipStr, ifaceName)) {
        if (ifaceName != "lo" && fallback.empty()) fallback = ipStr;
        continue;
      }
      freeifaddrs(ifaddr);
      return ipStr;
    }
  }

  freeifaddrs(ifaddr);

  // Prefer hostname resolution over link-local fallback.
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    std::string resolved = resolve_hostname(hostname);
    if (!resolved.empty()) return resolved;
  }
  return fallback;
}

static bool usable_cluster_ipv4(const std::string &ip, const std::string &iface) {
  if (iface == "lo") return false;
  if (ip.rfind("127.", 0) == 0) return false;
  if (ip.rfind("169.254.", 0) == 0) return false;
  return true;
}

static std::vector<std::string> get_local_ipv4_candidates() {
  std::vector<std::string> candidates;
  auto add_unique = [&](const std::string &ip) {
    if (ip.empty()) return;
    for (const auto &c : candidates) {
      if (c == ip) return;
    }
    candidates.push_back(ip);
  };

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0) {
    add_unique(resolve_hostname(hostname));
  }

  struct ifaddrs *ifaddr, *ifa;
  if (getifaddrs(&ifaddr) != -1) {
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET)
        continue;
      char ip[INET_ADDRSTRLEN];
      auto *sa = (struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &(sa->sin_addr), ip, INET_ADDRSTRLEN);
      std::string ipStr(ip);
      std::string ifaceName(ifa->ifa_name);
      if (usable_cluster_ipv4(ipStr, ifaceName)) add_unique(ipStr);
    }
    freeifaddrs(ifaddr);
  }

  add_unique(get_local_ip_addr());
  return candidates;
}

// DNS
std::string resolve_hostname(const std::string &hostname) {
  struct addrinfo hints {}, *res;
  hints.ai_family = AF_INET;  // IPv4 only
  hints.ai_socktype = SOCK_STREAM;

  // 解析主机名
  if (getaddrinfo(hostname.c_str(), nullptr, &hints, &res) != 0) {
    std::cerr << "getaddrinfo failed for hostname: " << hostname << std::endl;
    return "";
  }

  std::string result;
  for (struct addrinfo *p = res; p != nullptr; p = p->ai_next) {
    char ip[INET_ADDRSTRLEN];
    auto *addr = (struct sockaddr_in *)p->ai_addr;
    inet_ntop(AF_INET, &(addr->sin_addr), ip, INET_ADDRSTRLEN);

    // pass 127.0.0.1
    if (std::string(ip).find("127.") != 0) {
      result = ip;
      break;  
    }
  }

  freeaddrinfo(res);

  if (result.empty()) {
      std::cerr << "No non-loopback address found for hostname: " << hostname << std::endl;
  }

  return result;
}


// read ip_list
std::unordered_map<std::string, int> read_ip_map(const std::string &filename) {
  std::unordered_map<std::string, int> ipMap;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Fatal error: Can not open ip_list file: " << filename
              << std::endl;
    return ipMap;
  }

  // get first line of leader machine info. 
  std::string addr, port;
  std::getline(file, addr);
  std::getline(file, port);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string hostOrIP;
    int id;
    if (std::getline(iss, hostOrIP, '=') && iss >> id) {
      std::string ip = resolve_hostname(hostOrIP);
      if (!ip.empty()) {
        ipMap[ip] = id;
      } else {
        std::cerr << "WARN: Can not resolve" << hostOrIP << ", skip."
                  << std::endl;
      }
    }
  }

  file.close();
  return ipMap;
}

int get_machine_id(std::string config_file) {
  // const std::string configFile = "../scripts/ip_list.conf";

  auto ipMap = read_ip_map(config_file);

  if (ipMap.empty()) {
    std::cerr << "Error: ip_list file is empty or cannot resolve " << std::endl;
    return 1;
  }

  auto candidates = get_local_ipv4_candidates();
  if (candidates.empty()) {
    std::cerr << "Can not get local IP addr." << std::endl;
    return 1;
  }

  for (const auto &localIP : candidates) {
    auto it = ipMap.find(localIP);
    if (it != ipMap.end()) {
      std::cout << "Local machine IP addr: " << localIP << std::endl;
      std::cout << "Local machine ID: " << it->second << std::endl;
      return it->second;
    }
  }

  std::cout << "Error: local machine IP cannot find in list." << std::endl;
  std::cout << "Tried:";
  for (const auto &ip : candidates) std::cout << " " << ip;
  std::cout << std::endl;
  abort();
  return -1;
}


std::vector<std::string> get_machine_name(std::string config_file){
  std::vector<std::string> machine_name;
  std::ifstream file(config_file);

  if (!file.is_open()) {
    std::cerr << "Fatal error: Can not open ip_list file: " << config_file
              << std::endl;
    return machine_name;
  }

  // get first line of leader machine info. 
  std::string addr, port;
  std::getline(file, addr);
  std::getline(file, port);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string hostOrIP;
    int id;
    if (std::getline(iss, hostOrIP, '=') && iss >> id) {
      machine_name.push_back(hostOrIP);
    }
  }

  file.close();
  return machine_name;
}

