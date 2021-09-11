def make_plots(args, markers=True, numbers=False):
    import matplotlib.pyplot as plt
    import pandas as pd
    import pathlib

    df = pd.read_pickle(pathlib.Path(args.root).joinpath('results_dataframe.pkl'))
    df = df.sort_values('Loss')
    print(df)
    figs_dir = pathlib.Path(args.root).joinpath('PyFigs')
    figs_dir.mkdir(parents=True, exist_ok=True)
    if not figs_dir.exists():
        raise FileNotFoundError(f"{figs_dir} does not exist.")
    X_Keys = ['Loss']
    Y_Keys = ['Fitness', 'F1Score', 'MaxError', 'VolumeAccuracy']
    for k1 in X_Keys:
        for k2 in Y_Keys:
            title = f'{k1} vs. {k2}'
            f = plt.figure(figsize=(24, 13.5), dpi=80)
            plt.scatter(df[k1], df[k2])
            for i in range(len(df.index)):
                plt.text(df[k1][i], df[k2][i], str(i))
            plt.xlabel(k1)
            plt.ylabel(k2)
            plt.title(title)
            plt.savefig(figs_dir.joinpath(f"figure_{k1}_v_{k2}.png"))
            # plt.show()
            plt.close(f)


if __name__ == '__main__':
    class Foo(object):
        pass
    import os

    args = Foo()
    setattr(args, 'root', os.path.expanduser('~/Downloads'))

    make_plots(args)
