from functools import reduce


class DataField:
    def __init__(self, *, datamodel, data_source_name, field_name):
        self.field_name = field_name
        self.data_source_name = data_source_name
        self.datamodel = datamodel

    @staticmethod
    def reduce_by_keychain(target_dict, keychain):
        key_list = keychain.split('/')
        data = reduce(lambda x, y: x[y], key_list, target_dict)
        return data

    def get_data(self, shot_num):
        raise NotImplementedError

    def set_data(self, shot_num, data):
        raise NotImplementedError


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
        make_data_dict_pathchain = f'processed_{self.scale}_data/{self.data_source_name}/results/{shot_key}/'
        return make_data_dict_pathchain

    def get_data(self, shot_num):
        data_dict_pathchain = self.make_data_dict_pathchain(shot_num)
        shot_dict = self.reduce_by_keychain(target_dict=self.data_dict, keychain=data_dict_pathchain)
        data = shot_dict['field_name']
        return data

    def set_data(self, shot_num, data):
        data_dict_pathchain = self.make_data_dict_pathchain(shot_num)
        shot_dict = self.create_sub_dict(self.data_dict, data_dict_pathchain)
        shot_dict[self.field_name] = data
