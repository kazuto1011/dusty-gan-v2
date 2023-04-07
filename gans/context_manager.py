# adopted from
# https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py

from contextlib import ExitStack, contextmanager


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]

    return multi_contexts


def gradient_accumulation(num_accumulation, is_ddp, ddp_models):
    if is_ddp:
        # Grads are synchronized at the final round
        num_no_syncs = num_accumulation - 1
        head = [
            combine_contexts(map(lambda ddp: ddp.no_sync, ddp_models))
        ] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * num_accumulation

    for i, context in enumerate(contexts):
        with context():
            yield i
