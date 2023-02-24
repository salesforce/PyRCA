import os
import pandas as pd


def load_data(directory="/home/ywz/data/timeseries/warden_new", pod="eu44"):
    if isinstance(pod, (list, tuple)):
        dfs = [load_data(directory, pod=p) for p in pod]
        return pd.concat(dfs, axis=0)

    metrics = [
        "allocated_mb_per_interval_1p",
        "app_host_with_cpu_1p",
        "app_memory_allocation_1p",
        "app_pri_ram_1p",
        "apt_no_mq_1p",
        "avg_gc_usage_1p",
        "code_heap_nonprofiled_methods_1p",
        "concurrent_apex_error_1p",
        "conn_pool_waits_1p",
        "db_act_1p",
        "db_cpu_1p",
        "db_total_time_1p",
        "file_open_top_5_avg_1p",
        "gack_count_1p",
        # "ha_proxy_1p",
        "jetty_qtp_busy_threads_1p",
        "jvm_uptime_1p",
        "mem_promoted_1p",
        "old_gen_size_1p",
        "ping_status_1p",
        # "raid_iops_utlization_1p",
        "safepoint-time_1p",
        "successful_logins_1p",
        "thread_count_top_10_1p",
        "top_10_avg_cpu_1p",
        "trust_request_count_1p",
        "trust_request_nomq_count_1p",
        "young_pause_time_1p"
    ]

    df = pd.read_csv(os.path.join(directory, f"{pod}.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Timestamp"})
    df = df.set_index("Timestamp")
    df.columns = [col.replace(f"_{pod}", "") for col in df.columns]
    df = df[metrics]

    new_columns = []
    for c in df.columns:
        i = c.find("(")
        if i != -1:
            c = c[:i]
        new_columns.append(c.strip().replace(" ", "_").replace("%", ""))
    df.columns = new_columns

    new_df = None
    for metric in df.columns:
        x = df[[metric]].dropna()
        x.index = pd.to_datetime(x.index.values).floor('Min')
        new_df = x if new_df is None else new_df.join(x, how="inner")
    return new_df.dropna()
