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
    if (LogLevel::INFO < current_level) return;
    
    printf("[INFO] ");
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

void LOG_WARNING(const char* format, ...) {
    if (LogLevel::WARNING < current_level) return;
    
    printf("[WARNING] ");
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

void LOG_ERROR(const char* format, ...) {
    if (LogLevel::ERROR < current_level) return;
    
    printf("[ERROR] ");
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

void LOG_DEBUG(const char* format, ...) {
    if (LogLevel::DEBUG < current_level) return;
    
    printf("[DEBUG] ");
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}