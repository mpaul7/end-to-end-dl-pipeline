params = {
    "project": {
    "type": "dl",
    "project_home": ".",
    "model_type": ["cnn"],
        "run_name": "cnn_2",
    },

    "features": {
        "cn_stat_feature_length": 0,
        "sequence_length": 30,
        "seq_packet_feature": [],
        "cnn_stat_feature": "",
        "stat_features": [],        
    },

    "labels": {
        "target_column": "refined_app_label",
        "target_labels": ["Discord", "Telegram", "Microsoft Teams", "Whatsapp", "Facebook Messenger", "Signal"]
    },

    "data_split": {
        "test_size": 0.2,
        "validation_size": 0.2
    }
    
}