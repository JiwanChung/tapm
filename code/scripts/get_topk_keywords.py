from common import load_data, parse, run, GetWords


def main():
    args = parse()
    run(args, GetTopkWords(), 'top')


class GetTopkWords(GetWords):
    def __init__(self):
        super(GetTopkWords, self).__init__()

    def __call__(self, sent):
        return super().__call__(sent)


if __name__ == "__main__":
    main()
