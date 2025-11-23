#pragma once
#include <string>

enum class LogLevel { INFO, WARNING, ERROR, DEBUG };

void LOG_INIT();
void LOG_SET_LEVEL(LogLevel level);
void LOG(LogLevel level, const char* format, ...);
void LOG_INFO(const char* format, ...);
void LOG_WARNING(const char* format, ...);
void LOG_ERROR(const char* format, ...);
void LOG_DEBUG(const char* format, ...);