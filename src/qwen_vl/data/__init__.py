import re

# Placeholder roots. Replace these with the actual locations on your machine.
ANNOTATIONS_ROOT = "/path/to/stream3d_dataset_public/train"
CA1M_DATA_ROOT = "/path/to/datasets/ca1m"
SCANNET_DATA_ROOT = "/path/to/datasets/scannet_v2"
SCANNETPP_DATA_ROOT = "/path/to/datasets/scannetpp"


CA1M_Ego_Motion_Estimation = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/ca1m_ego_motion_estimation_train.json",
    "data_path": CA1M_DATA_ROOT,
    "tag": "3d",
}

ScanNet_Ego_Motion_Estimation = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannet_ego_motion_estimation_train.json",
    "data_path": SCANNET_DATA_ROOT,
    "tag": "3d",
}

ScanNet_Environment_Measurement = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannet_environment_measurement_train.json",
    "data_path": SCANNET_DATA_ROOT,
    "tag": "3d",
}

ScanNet_Object_Attributes = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannet_object_attributes_train.json",
    "data_path": SCANNET_DATA_ROOT,
    "tag": "3d",
}

ScanNet_Object_Camera_Relationship = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannet_object_camera_relationship_train.json",
    "data_path": SCANNET_DATA_ROOT,
    "tag": "3d",
}

ScanNet_Object_Chronology = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannet_object_chronology_train.json",
    "data_path": SCANNET_DATA_ROOT,
    "tag": "3d",
}

ScanNetpp_Ego_Motion_Estimation = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannetpp_ego_motion_estimation_train.json",
    "data_path": SCANNETPP_DATA_ROOT,
    "tag": "3d",
}

ScanNetpp_Environment_Measurement = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannetpp_environment_measurement_train.json",
    "data_path": SCANNETPP_DATA_ROOT,
    "tag": "3d",
}

ScanNetpp_Object_Camera_Relationship = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannetpp_object_camera_relationship_train.json",
    "data_path": SCANNETPP_DATA_ROOT,
    "tag": "3d",
}

ScanNetpp_Object_Chronology = {
    "annotation_path": f"{ANNOTATIONS_ROOT}/scannetpp_object_chronology_train.json",
    "data_path": SCANNETPP_DATA_ROOT,
    "tag": "3d",
}

data_dict = {
    "ca1m-ego_motion_estimation": CA1M_Ego_Motion_Estimation,
    "scannet-ego_motion_estimation": ScanNet_Ego_Motion_Estimation,
    "scannet-environment_measurement": ScanNet_Environment_Measurement,
    "scannet-object_attributes": ScanNet_Object_Attributes,
    "scannet-object_camera_relationship": ScanNet_Object_Camera_Relationship,
    "scannet-object_chronology": ScanNet_Object_Chronology,
    "scannetpp-ego_motion_estimation": ScanNetpp_Ego_Motion_Estimation,
    "scannetpp-environment_measurement": ScanNetpp_Environment_Measurement,
    "scannetpp-object_camera_relationship": ScanNetpp_Object_Camera_Relationship,
    "scannetpp-object_chronology": ScanNetpp_Object_Chronology,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
