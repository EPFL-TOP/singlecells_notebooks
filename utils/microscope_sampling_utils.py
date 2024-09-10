import nd2
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def experiment_sampling_t(nd_path):
    """
    Extracts and returns the sampling times from the metadata stored in an ND2 file.

    This function reads an ND2 (Nikon ND2) file and retrieves various time-related parameters
    associated with the experiment, such as the expected time between frames,
    the average frame period difference, the minimum frame period difference,
    and the maximum frame period difference.

    Parameters:
    -----------
    nd_path : str
        The file path of the ND2 containing the metadata.

    Returns:
    --------
    tuple
        A tuple containing the following values (in seconds):
        - expected_time (float): The time between frames in the experiment as selected in the acquisition setup.
        - avg_exp_time (float): The measured average time difference between frames.
        - min_exp_time (float): The measured minimum time difference between frames.
        - max_exp_time (float): The measured maximum time difference between frames.

    Raises:
    -------
    FileNotFoundError
        If the specified ND2 file does not exist or cannot be opened.

    Notes:
    ------
    - The function uses the `nd2` module to access the ND2 file and read its parameters (https://pypi.org/project/nd2/).
    - The function has been tested with nd2 module version 0.10.1.
    - The values extracted from the metadata (in milliseconds) are converted to seconds by dividing by 1000.

    """
    try:
        with nd2.ND2File(nd_path) as ndfile:
            expected_time = ndfile.experiment[0].parameters.periodMs / 1000
            avg_exp_time = ndfile.experiment[0].parameters.periodDiff.avg / 1000
            min_exp_time = ndfile.experiment[0].parameters.periodDiff.min / 1000
            max_exp_time = ndfile.experiment[0].parameters.periodDiff.max / 1000

        return expected_time, avg_exp_time, min_exp_time, max_exp_time
    except FileNotFoundError:
        print('Not a valid nd2 file')


def position_time_stamps(nd_path_split):
    """
    Extracts time stamps from an ND2 file and calculates time differences based on expected sampling times.

    This function reads an ND2 (Nikon ND2) file containing one single position, extracts time stamps from each event,
    and performs the following operations:
    1. Retrieves the actual time stamps for each event.
    2. Calculates the expected time stamps based on the sampling period.
    3. Computes the difference between the actual and expected time stamps (with and without offset).
    4. Calculates the difference between consecutive time stamps.

    Parameters:
    -----------
    nd_path_split : str
        The file path of the ND2 file to be analyzed.

    Returns:
    --------
    tuple
        A tuple containing:
        - t_values_arr (np.ndarray): The array of actual time stamps (in seconds) for each event.
        - t_values_expected_diff_woffset (np.ndarray): The array of differences between the actual
          and expected time stamps, adjusted by removing the offset of the first time stamp.
        - t_values_succ_diff (np.ndarray): The array of successive differences between the time stamp differences.

    Notes:
    ------
    - The function uses the `nd2` module to access the ND2 file and read its parameters (https://pypi.org/project/nd2/).
    - The function has been tested with nd2 module version 0.10.1.
    - The expected input ND2 file is supposed to contain a single-position file coming from a multipoint file.
    - Expected time stamps are calculated using the expected sampling period (in seconds), which is obtained
      from the experiment parameters (`periodMs`).
    - Time differences are computed as:
        - `t_values_expected_diff`: Actual time stamp minus expected time stamp.
        - `t_values_expected_diff_woffset`: `t_values_expected_diff` with the offset of the first time stamp removed.
        - `t_values_succ_diff`: Successive differences between elements of `t_values_expected_diff`.
    """
    with nd2.ND2File(nd_path_split) as ndfile:
        expected_sampling_time = ndfile.experiment[0].parameters.periodMs / 1000
        t_values = []
        for time_stamp in range(len(ndfile.events())):
            t_values.append(ndfile.events()[time_stamp]['Time [s]'])
    t_values_arr = np.asarray(t_values)

    aux_arr = np.arange(0, len(t_values_arr), 1)
    aux_arr = aux_arr * expected_sampling_time
    t_values_expected_diff = t_values_arr - aux_arr
    t_values_expected_diff_woffset = t_values_expected_diff - t_values_arr[0]
    t_values_succ_diff = np.diff(t_values_expected_diff)
    return t_values_arr, t_values_expected_diff_woffset, t_values_succ_diff


def position_time_stamps_batch(root_directory):
    subdirectory_l = []
    for subdirectory in pathlib.Path(root_directory).iterdir():
        if subdirectory.is_dir():
            path_l = []
            for file in subdirectory.iterdir():
                if '.nd2' in file.as_posix():
                    path_l.append(file)
            if path_l:
                subdirectory_l.append(subdirectory)

    for subdirectory in subdirectory_l:
        name = subdirectory.name
        path_l = [file for file in subdirectory.iterdir() if '.nd2' in file.as_posix()]
        avg_t_values_exp = np.mean(np.asarray([position_time_stamps(p.as_posix())[1] for p in path_l]), axis=0)
        avg_t_values_succ_diff = np.mean(np.asarray([position_time_stamps(p.as_posix())[2] for p in path_l]), axis=0)

        # Plot
        plt.close('All')
        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
        fig.suptitle(name, fontsize=16)

        # avg_t_values_exp subplot
        axes[0].plot(avg_t_values_exp)
        axes[0].set_title('Average deviation from expected value')
        axes[0].set_xlabel('Time frame')
        axes[0].set_ylabel('Deviation (s)')

        # avg_t_values_succ_diff
        axes[1].plot(avg_t_values_succ_diff)
        axes[1].set_title('Average successive differences of deviations')
        axes[1].set_xlabel('Time frame')
        axes[1].set_ylabel('Deviation (s)')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pass
