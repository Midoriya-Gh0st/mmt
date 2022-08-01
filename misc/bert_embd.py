import pickle


def load_id(f):
    file = f'data/ids/ids_{f}.pkl'
    return list(pickle.load(open(file, 'rb')))


ids = load_id('2')[1:]

