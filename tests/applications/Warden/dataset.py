from pyrca.applications.WardenAIOps.dataset import load_data


def test():
    df = load_data(
        directory="/home/ywz/data/timeseries/warden_new",
        pod=["eu44", "na115", "na116", "na136", "na211"]
    )

    print(df.columns)
    print(df)


if __name__ == "__main__":
    test()
