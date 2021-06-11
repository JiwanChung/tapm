from pathlib import Path


def load_result(path):
    res = {}
    key_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split('\t')
            key = line[0]
            ans = line[-1].strip()
            res[key] = ans
            key_list.append(key)
    return res, key_list


def main():
    path = '~/projects/lsmdc/data/LSMDC/task2/test_results.csv'
    pathok = '~/projects/lsmdc/data/LSMDC/task2/test_textonly_results.csv'
    path = Path(path).expanduser()
    pathok = Path(pathok).expanduser()

    data, order = load_result(path)
    dataok, orderok = load_result(pathok)

    if set(data.keys()) != set(dataok.keys()):
        print('key difference')
        import ipdb; ipdb.set_trace()  # XXX DEBUG

    if order != orderok:
        print('order difference')
        import ipdb; ipdb.set_trace()  # XXX DEBUG

    for key in dataok.keys():
        d = data[key]
        dok = dataok[key]

        if len(d.split(',')) != len(dok.split(',')):
            print('wrong blank num')
            import ipdb; ipdb.set_trace()  # XXX DEBUG


if __name__ == "__main__":
    main()
