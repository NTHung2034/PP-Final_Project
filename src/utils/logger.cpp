#include "utils/logger.h"
#include <cstdio>
#include <cstdarg>
#include <chrono>

static LogLevel current_level = LogLevel::INFO;

void LOG_INIT() {
    // Initialize logging system
    current_level = LogLevel::INFO;
}

void LOG_SET_LEVEL(LogLevel level) {
    current_level = level;
}

void LOG(LogLevel level, const char* format, ...) {
    if (level < current_level) return;
    
    const char* level_str[] = {"INFO", "WARNING", "ERROR", "DEBUG"};
    
    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    // Print header
    printf("[%s] ", level_str[static_cast<int>(level)]);
    
    // Print message
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

void LOG_INFO(const char* format, ...) {
    va_list args;
    va_start(args, format);
    LOG(LogLevel::INFO, format, args);
    va_end(args);
}

// Similar implementations for WARNING, ERROR, DEBUG...