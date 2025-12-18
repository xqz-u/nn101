#ifndef LOGS_H
#define LOGS_H

#include <stdio.h>

typedef struct Logs Logs;
struct Logs {
    FILE* train_logs;
    FILE* train_batch_logs;
    FILE* eval_logs;
};

bool initLogs(Logs* logs,
              const char* train_logs_path,
              const char* train_batch_logs_path,
              const char* eval_logs_path);
void destroyLogs(Logs* logs);

#endif  // UTILS_h
