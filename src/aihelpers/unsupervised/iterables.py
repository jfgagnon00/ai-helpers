from itertools import islice


def chunkify_iterables(iterables, chunk_size=None):
    if chunk_size is None:
        yield from iterables
    else:
        it = iter(iterables)
        yield from iter(lambda: list(islice(it, chunk_size)), [])