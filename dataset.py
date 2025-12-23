from datasets.mhri import MHRI

def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MHRI'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'MHRI':
        training_data = MHRI(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform)
    return training_data

def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MHRI'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'MHRI':
        validation_data = MHRI(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform)
    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MHRI'], print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'MHRI':
        test_data = MHRIS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return test_data
