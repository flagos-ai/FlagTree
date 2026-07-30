#include <iostream>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int main() {
    struct perf_event_attr pe;
    std::memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    // Test 1: Cycles
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    int fd_cycles = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd_cycles == -1) {
        std::cerr << "Error opening CYCLES counter: " << std::strerror(errno) << " (errno=" << errno << ")" << std::endl;
    } else {
        std::cout << "Successfully opened CYCLES counter (fd=" << fd_cycles << ")" << std::endl;
        close(fd_cycles);
    }

    // Test 2: Instructions
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    int fd_inst = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd_inst == -1) {
        std::cerr << "Error opening INSTRUCTIONS counter: " << std::strerror(errno) << " (errno=" << errno << ")" << std::endl;
    } else {
        std::cout << "Successfully opened INSTRUCTIONS counter (fd=" << fd_inst << ")" << std::endl;
        close(fd_inst);
    }

    return 0;
}
