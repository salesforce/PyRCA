from pyrca.applications.SDDB.dataset import \
    preprocess, load_data


def test_preprocess():
    directory = "/home/ywz/data/timeseries/SDDB_new"
    filenames = [
        "sddb_db_metrics.dat.4",
        "sddb_db_metrics.dat.3",
        "sddb_db_metrics.dat.2",
        "sddb_db_metrics.dat.1",
    ]
    preprocess(directory, filenames)


def test_load_data():
    directory = "/home/ywz/data/timeseries/SDDB_new"
    df, _, _ = load_data(
        directory, date_range=("2022-10-01", "2022-10-31"))
    print(df.head())
    print(df.shape)
    print(df.columns)


if __name__ == "__main__":
    test_load_data()
