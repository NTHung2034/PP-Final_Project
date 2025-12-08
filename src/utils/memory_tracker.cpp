#include "utils/memory_tracker.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <unistd.h>
#include <sys/resource.h>
#endif

namespace MemoryTracker
{

    static size_t peak_memory = 0;
    static char format_buffer[64];

    void init()
    {
        peak_memory = 0;
    }

    size_t get_rss()
    {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        {
            return pmc.WorkingSetSize;
        }
        return 0;
#else
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss * 1024; // Convert KB to bytes on Linux
#endif
    }

    size_t get_virtual_memory()
    {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        {
            return pmc.PagefileUsage;
        }
        return 0;
#else
        FILE *file = fopen("/proc/self/status", "r");
        if (!file)
            return 0;

        char line[128];
        size_t vmsize = 0;

        while (fgets(line, sizeof(line), file))
        {
            if (strncmp(line, "VmSize:", 7) == 0)
            {
                sscanf(line + 7, "%zu", &vmsize);
                vmsize *= 1024; // Convert KB to bytes
                break;
            }
        }

        fclose(file);
        return vmsize;
#endif
    }

    size_t get_current_usage()
    {
        size_t current = get_rss();
        peak_memory = std::max(peak_memory, current);
        return current;
    }

    size_t get_peak_usage()
    {
        return peak_memory;
    }

    void reset_peak()
    {
        peak_memory = 0;
    }

    const char *format_bytes(size_t bytes)
    {
        const char *units[] = {"B", "KB", "MB", "GB"};
        int unit_idx = 0;
        double size = static_cast<double>(bytes);

        while (size >= 1024.0 && unit_idx < 3)
        {
            size /= 1024.0;
            unit_idx++;
        }

        snprintf(format_buffer, sizeof(format_buffer), "%.2f %s", size, units[unit_idx]);
        return format_buffer;
    }

} // namespace MemoryTracker
