import numpy as np
import re
import mne
import neo.io

filepath = r'C:\Users\deudon\Desktop\Epifar\_Scripts\micMac\exampleData\SAB ENC 7 550ms.edf'
filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\data_clean\SC23-stimulations-1_clean_raw.fif'
filepath = r'C:\Users\deudon\Desktop\TestFiles\COG_024_EEG_95.TRC'
filepath = r'C:\Users\deudon\Desktop\Epifar\_Scripts\micMac\exampleData\resync_ex\20181002-150016-001.ns5'



def io_eeg_to_mne(filepath, read_data=True):
    """ Import an EEG file to MNE as a Raw instance.
    Supported formats : 'edf', 'eeg', 'trc'

    Parameters
    ----------
    filepath : str
        EEG filepath
    read_data : bool (default: True)
        If True, read the data

    Returns
    -------
    mne_raw : MNE RAW instance
        Output MNE structure
    ch_names : list
        Channel names

    """
    mne_raw = []
    possible_ext = ['.edf', '.fif', '.trc', '.ns5', '.nsx']
    file_ext = re.search('\.\w+$', filepath)
    if file_ext:
        file_ext = file_ext[0].lower()
    else:
        raise ValueError('Could not detect file extension of file {}'.format(filepath))
    if file_ext not in possible_ext:
        raise ValueError('The file {} has not a supported extension. Extensions must be in {}'.format(filepath,
                                                                                                      possible_ext))
    if file_ext in ['.edf']:
        try:
            mne_raw = mne.io.read_raw_edf(filepath, preload=read_data)
        except:
            mne_raw = mne.io.read_raw_edf(filepath, preload=True)
        ch_names = mne_raw.ch_names
    elif file_ext == '.fif':
        try:
            mne_raw = mne.io.read_raw_fif(filepath, preload=read_data)
        except:
            mne_raw = mne.io.read_raw_fif(filepath, preload=True)
        ch_names = mne_raw.ch_names
    elif file_ext == '.trc':
        trc_reader = neo.io.MicromedIO(filename=filepath)
        header = trc_reader.header
        ch_names = [header['signal_channels'][i][0] for i in range(trc_reader.signal_channels_count())]
        if read_data:
            bl = trc_reader.read(lazy=False)[0]
            seg = bl.segments[0]
            n_pnts, n_chan = len(seg.analogsignals[0]), len(seg.analogsignals)
            data = np.zeros((n_chan, n_pnts), dtype=float)
            for i, asig in enumerate(seg.analogsignals):
                # We need the ravel() here because Neo < 0.5 gave 1D, Neo 0.5 gives 2D (but still a single channel).
                data[i, :] = asig.magnitude.ravel()
            sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            mne_raw = mne.io.RawArray(data, info)
    elif file_ext in ['.ns5', 'nsx']:
        nsx_reader = neo.io.BlackrockIO(filename=filepath)
        header = nsx_reader.header
        ch_names = [header['signal_channels'][i][0] for i in range(nsx_reader.signal_channels_count())]
        if read_data:
            bl = nsx_reader.read(lazy=False)[0]
            seg = bl.segments[0]
            n_pnts, n_chan = len(seg.analogsignals[0]), len(seg.analogsignals)
            data = np.zeros((n_chan, n_pnts), dtype=float)
            for i, asig in enumerate(seg.analogsignals):
                # We need the ravel() here because Neo < 0.5 gave 1D, Neo 0.5 gives
                # 2D (but still a single channel).
                data[i, :] = asig.magnitude.ravel()
            sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            mne_raw = mne.io.RawArray(data, info)

    return mne_raw, ch_names

