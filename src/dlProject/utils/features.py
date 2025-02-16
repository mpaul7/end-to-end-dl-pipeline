stat_features_twc = ["pkt_fwd_count", 
    "pl_fwd_count",
    "pl_len_fwd_mean", 
    "pl_len_fwd_stdev",
    "pl_len_fwd_total", 
    "pl_len_fwd_min", 
    "pl_len_fwd_max",
    "pkt_len_fwd_mean", 
    "pkt_len_fwd_stdev", 
    "pkt_len_fwd_total",
    "pkt_len_fwd_min", 
    "pkt_len_fwd_max", 
    "iat_fwd_mean", 
    "iat_fwd_stdev",
    "iat_fwd_total", 
    "iat_fwd_min", 
    "iat_fwd_max", 
    "pkt_bwd_count",
    "pl_bwd_count", 
    "last_timestamp_bwd", 
    "pl_len_bwd_mean",
    "pl_len_bwd_stdev", 
    "pl_len_bwd_total", 
    "pl_len_bwd_min",
    "pl_len_bwd_max", 
    "pkt_len_bwd_mean", 
    "pkt_len_bwd_stdev",
    "pkt_len_bwd_total", 
    "pkt_len_bwd_min", 
    "pkt_len_bwd_max",
    "iat_bwd_mean", 
    "iat_bwd_stdev", 
    "iat_bwd_total", 
    "iat_bwd_min",
    "iat_bwd_max"
]

stat_features_twc_test = ["pl_len_fwd_min", "pl_len_fwd_mean", "pl_len_fwd_max", 
"pl_len_fwd_stdev", "pl_len_bwd_min", "pl_len_bwd_mean", 
"pl_len_bwd_max", "pl_len_bwd_stdev", "pkt_len_fwd_min", 
"pkt_len_fwd_mean", "pkt_len_fwd_max", "pkt_len_fwd_stdev", 
"pkt_len_bwd_min", "pkt_len_bwd_mean", "pkt_len_bwd_max", 
"pkt_len_bwd_stdev"] 

seq_packet_feature = ["splt_ps", "splt_piat", "splt_direction"]

stat_features_nfs = [
    'bidirectional_duration_ms', 'bidirectional_packets',
    'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets',
    'src2dst_bytes', 'dst2src_duration_ms', 'dst2src_packets',
    'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps',
    'bidirectional_stddev_ps', 'bidirectional_max_ps', 'src2dst_min_ps',
    'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
    'dst2src_max_ps', 'bidirectional_min_piat_ms',
    'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
    'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
    'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    'dst2src_max_piat_ms', 'bidirectional_syn_packets',
    'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    'bidirectional_urg_packets', 'bidirectional_ack_packets',
    'bidirectional_psh_packets', 'bidirectional_rst_packets',
    'bidirectional_fin_packets', 'src2dst_syn_packets',
    'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
    'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
    'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
    'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets',
    'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets'
]

target_column = "refined_app_label"
target_labels = ["Discord", "Telegram", "Microsoft Teams", "Whatsapp", "Facebook Messenger", "Signal"]


