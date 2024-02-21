import pickle

with open('nuscenes_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)
    l = []
    for data_point in data['data_list']:
        l.append(data_point['token'])


print(l)
