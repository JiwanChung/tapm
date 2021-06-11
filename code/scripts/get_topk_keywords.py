from .common import load_data, run, GetWords


def main():
    run(GetTopkWords(), 'top')


class GetTopkWords(GetWords):
    def __init__(self):
        super(GetTopkWords, self).__init__()

    def __call__(self, sent):
        return super().__call__(sent)
