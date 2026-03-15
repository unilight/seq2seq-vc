import h5py
import argparse


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("rename stats file")
    )
    parser.add_argument(
        "name",
        type=str,
    )
    args = parser.parse_args()

    hdf5_name = args.name
    hdf5_file = h5py.File(hdf5_name, "r+")
    mean = hdf5_file["mean"][()]
    scale = hdf5_file["scale"][()]
    hdf5_file.create_dataset("mel_mean", data=mean); hdf5_file.flush()
    hdf5_file.create_dataset("mel_scale", data=scale); hdf5_file.flush()
    hdf5_file.close()


if __name__ == "__main__":
    main()
