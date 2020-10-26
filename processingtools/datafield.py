from functools import reduce
from pathlib import Path
import h5py
from .datatools import InputParamLogger


class DataField(InputParamLogger):
    def __init__(self, *, datamodel, data_source_name, field_name):
        self.field_name = field_name
        self.data_source_name = data_source_name
        self.datamodel = datamodel
        self.datamodel.add_datafield(self)

    @staticmethod
    def reduce_by_keychain(data_source, keychain):
        key_list = keychain.split('/')
        data = reduce(lambda x, y: x[y], key_list, data_source)
        return data

    def get_data(self, shot_num):
        raise NotImplementedError

    def set_data(self, shot_num, data):
        raise NotImplementedError


class H5DataField(DataField):
    """
    mode = 'raw' or 'processed'
    """
    def __init__(self, *, datamodel, data_source_name, field_name, file_prefix, h5_subpath, mode):
        super(H5DataField, self).__init__(datamodel=datamodel, data_source_name=data_source_name,
                                          field_name=field_name)
        self.file_prefix = file_prefix
        self.h5_subpath = h5_subpath
        self.mode = mode
        self.daily_path = datamodel.daily_path
        self.run_name = datamodel.run_name
        if self.mode == 'raw':
            self.data_source_path = Path(self.daily_path, 'data', self.run_name, self.data_source_name)
        elif self.mode == 'processed':
            self.data_source_path = Path(self.daily_path, 'analysis', self.run_name, self.data_source_name)

        subpath_list = (self.h5_subpath.split('/'))
        self.subgroup_list = subpath_list[:-1]
        self.dataset_name = subpath_list[-1]

    def get_containing_group(self, h5_file):
        if len(self.subgroup_list) > 0:
            group_chain = '/'.join(self.subgroup_list)
            h5_file.require_group(group_chain)
            return h5_file[group_chain]
        else:
            return h5_file

    def get_data_file_path(self, shot_num):
        file_name = f'{self.file_prefix}_{shot_num:05d}.h5'
        file_path = Path(self.data_source_path, file_name)
        return file_path

    def get_data(self, shot_num):
        file_path = self.get_data_file_path(shot_num)
        data_file = h5py.File(file_path, 'r')
        containing_group = self.get_containing_group(data_file)
        data = containing_group[self.dataset_name]
        return data

    def set_data(self, shot_num, data):
        if self.mode == 'raw':
            print('Cannot set data in RawH5DataField - this is raw data.')
        elif self.mode == 'processed':
            file_path = self.get_data_file_path(shot_num)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data_file = h5py.File(file_path, 'a')
            containing_group = self.get_containing_group(data_file)
            try:
                del containing_group[self.dataset_name]
            except (OSError, KeyError):
                pass
            containing_group.create_dataset(self.dataset_name, data=data)

    def set_attr(self, shot_num, attr_name, attr):
        if self.mode == 'raw':
            print('Cannot set data in RawH5DataField - this is raw data.')
        elif self.mode == 'processed':
            file_path = self.get_data_file_path(shot_num)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data_file = h5py.File(file_path, 'a')
            containing_group = self.get_containing_group(data_file)
            dataset = containing_group[self.dataset_name]
            dataset.attrs[attr_name] = attr


class GageRawDataField(H5DataField):
    def __init__(self, *, datamodel, data_source_name, field_name, file_prefix, h5_subpath):
        super(GageRawDataField, self).__init__(datamodel=datamodel, data_source_name=data_source_name,
                                               field_name=field_name, file_prefix=file_prefix, h5_subpath=h5_subpath,
                                               mode='raw')
        self.channel_name, self.segment_name = self.h5_subpath.split('/')

    def get_data(self, shot_num):
        data = super(GageRawDataField, self).get_data(shot_num)
        file_path = self.get_data_file_path(shot_num)
        data = self.scale_gagescope_data(data, file_path)
        return data

    def scale_gagescope_data(self, data, file_path):
        h5 = h5py.File(file_path, 'r')
        channel_data = h5[self.channel_name]
        sample_offset = channel_data.attrs['sample_offset']
        sample_res = channel_data.attrs['sample_res']
        sample_range = channel_data.attrs['input_range']
        offset_v = channel_data.attrs['dc_offset']
        scaled_data = ((sample_offset - data) / sample_res) * (sample_range / 2000.0) + offset_v
        # dt = data.attrs['dx']
        return scaled_data


class DataDictField(DataField):
    """
    scale is either 'shot' or 'point'
    """
    def __init__(self, *, datamodel, field_name, data_source_name, scale):
        super(DataDictField, self).__init__(datamodel=datamodel, data_source_name=data_source_name,
                                            field_name=field_name, )
        self.data_dict = self.datamodel.data_dict
        self.scale = scale

    @staticmethod
    def create_sub_dict(parent_dict, child_dict_keychain):
        children_dict_names = child_dict_keychain.split('/')
        curr_dict = parent_dict
        for child_dict_name in children_dict_names:
            if child_dict_name not in curr_dict:
                curr_dict[child_dict_name] = dict()
            curr_dict = curr_dict[child_dict_name]
        return curr_dict

    def make_data_dict_pathchain(self, shot_num):
        shot_key = f'shot-{shot_num:d}'
        make_data_dict_pathchain = f'{self.scale}_data/{self.data_source_name}{shot_key}/'
        return make_data_dict_pathchain

    def get_data(self, shot_num):
        data_dict_pathchain = self.make_data_dict_pathchain(shot_num)
        shot_dict = self.reduce_by_keychain(data_source=self.data_dict, keychain=data_dict_pathchain)
        data = shot_dict[self.field_name]
        return data

    def set_data(self, shot_num, data):
        data_dict_pathchain = self.make_data_dict_pathchain(shot_num)
        shot_dict = self.create_sub_dict(self.data_dict, data_dict_pathchain)
        shot_dict[self.field_name] = data
