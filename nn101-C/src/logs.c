#include "logs.h"

bool initLogs(Logs* logs,
              const char* train_logs_path,
              const char* train_batch_logs_path,
              const char* eval_logs_path) {
    if (logs == nullptr) {
        return false;
    }

    if ((logs->train_logs = fopen(train_logs_path, "w")) == nullptr) {
        printf("Error opening log file '%s'\n", train_logs_path);
        return false;
    }
    fprintf(logs->train_logs, "epoch,loss,time\n");
    fflush(logs->train_logs);

    if ((logs->train_batch_logs = fopen(train_batch_logs_path, "w")) ==
        nullptr) {
        printf("Error opening log file '%s'\n", train_batch_logs_path);
        return false;
    }
    fprintf(logs->train_batch_logs, "batch,loss\n");
    fflush(logs->train_batch_logs);

    if ((logs->eval_logs = fopen(eval_logs_path, "w")) == nullptr) {
        printf("Error opening log file '%s'\n", eval_logs_path);
        return false;
    }
    fprintf(logs->eval_logs, "epoch,accuracy,time\n");
    fflush(logs->eval_logs);

    return true;
}

void destroyLogs(Logs* logs) {
    if (logs == nullptr) {
        return;
    }
    if (logs->train_logs) {
        fclose(logs->train_logs);
    }
    if (logs->train_batch_logs) {
        fclose(logs->train_batch_logs);
    }
    if (logs->eval_logs) {
        fclose(logs->eval_logs);
    }
}
