from collections import deque

import webdataset as wds
from tqdm import tqdm


def sample_seq(data, seq_len=5, burn_in=0):
    seq_len = seq_len + burn_in
    # concate in a deque, then yield.
    list_of_tuples = deque(maxlen=seq_len)
    for d in data:
        list_of_tuples.append(d)
        if len(list_of_tuples) == seq_len:
            tuple_of_lists = tuple(zip(*list_of_tuples))
            keys = tuple_of_lists[2]
            if is_valid(keys, seq_len):
                # states    tuple_of_lists[0]
                # ard       tuple_of_lists[1]
                # keys      tuple_of_lists[2]
                yield tuple_of_lists[2]


def is_valid(seq, seq_len):
    """Checks the sequence is not crossing shard boundaries.
    We do this by checking the idxs are consistent.
    """
    diff = int(seq[-1].split(":")[-1]) - int(seq[0].split(":")[-1])
    print(diff)
    return diff == (seq_len - 1)


def main():
    path = "./fold0/{asterix,breakout,enduro,mspacman,seaquest,spaceinvaders}_{0000..0160}.tar"
    dset = (
        wds.WebDataset(path, shardshuffle=True)
        .decode("rgb8")
        .to_tuple("state.png", "ard.msg", "__key__")
        .map_tuple(None, None, None)
        .compose(sample_seq)
        # .shuffle(1000)
        .batched(2)
    )

    for batch in dset:
        input()

        batch = list(zip(*batch))
        for seq in batch:
            idxs = [int(k.split("_")[-1].split(":")[-1]) for k in seq]
            pfix = set(["_".join(k.split("_")[:-1]) for k in seq])
            print(pfix, idxs)


if __name__ == "__main__":
    main()
