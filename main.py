def train_partial(*args, **kwargs):
    import train_and_test
    train_and_test.train(*args, **kwargs, dataset_type="partial")


def train_example(*args, **kwargs):
    import train_and_test
    train_and_test.train(*args, **kwargs, dataset_type="example")


if __name__ == '__main__':
    train_partial(epochs=1000, restore_last=False, progress_per_step=100)
    # train_example(epochs=1000, restore_last=False, progress_per_step=100)
