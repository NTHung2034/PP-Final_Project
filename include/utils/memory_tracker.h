#pragma once
#include <cstddef>

namespace MemoryTracker
{

    // Initialize memory tracking
    void init();

    // Get current memory usage in bytes
    size_t get_current_usage();

    // Get peak memory usage in bytes
    size_t get_peak_usage();

    // Reset peak memory tracking
    void reset_peak();

    // Get current RSS (Resident Set Size) in bytes
    size_t get_rss();

    // Get current virtual memory usage in bytes
    size_t get_virtual_memory();

    // Format bytes to human-readable string
    const char *format_bytes(size_t bytes);
}
