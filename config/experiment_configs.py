EXPERIMENT_TASK_INFO = {
    'nyuv2': {
        'channel': [13, 1, 3],
        'task_slices': [slice(0, 13), slice(13, 14), slice(14, None)],
        'tasks': ['semantic_segmentation', 'depth_estimation', 'normal_estimation'],
        'task_num': 3,
    },
    'bdd100k': {    # TODO: values are placeholders, update with actual values
        'channel': [12, 1, 3],
        'task_slices': [slice(0, 12), slice(12, 13), slice(13, None)],
        'tasks': ['semantic_segmentation', 'depth_estimation', 'normal_estimation'],
        'task_num': 3,
    },
}