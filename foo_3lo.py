from evobench.discrete import Trap
from tqdm.auto import tqdm
from hell.linkage import test_lo3


def main():
    benchmark = Trap(blocks=[4, 4, 4], verbose=1)
    results = test_lo3(benchmark, n_samples=10)
    foo = 2


if __name__ == "__main__":
    main()
