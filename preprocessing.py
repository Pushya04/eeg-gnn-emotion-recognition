# Importing python Library
import mne, os, time, pickle, warnings, copy, sys, shutil, argparse
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal
from tqdm import trange
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps  # Updated import
from mne.preprocessing import ICA
from mne.filter import filter_data as bandpass_filter
warnings.filterwarnings('ignore')

def SignalPreProcess(eeg_rawdata, subject_path):
    '''
    eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
    subject_path: Path to save ICA component scores graph
    return: list of segmented filtered EEG data of shape (n_segments, n_channels, segment_length)
    '''
    assert eeg_rawdata.shape[0] == 32
    eeg_rawdata = np.array(eeg_rawdata)
  
    info = mne.create_info(ch_names=channels, ch_types=['eeg' for _ in range(32)], sfreq=128, verbose=False)
    raw_data = mne.io.RawArray(eeg_rawdata, info, verbose=False)  # create MNE raw file
    # Bandpass filter of 4 Hz to 48 Hz
    raw_data.load_data(verbose=False).filter(l_freq=4, h_freq=48, method='fir', verbose=False)
    # raw_data.plot()
    
    try:
        # FAST-ICA with 31 number of components
        ica = ICA(n_components=None, random_state=97, verbose=False)
        ica.fit(raw_data)  # fit the data into ica
        # Take Fp1 channel as the reference channel and find the ICA score to choose artifacts score.
        eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name='Fp1', verbose=None)
        
        # Plot and save ICA component scores
        ica_plot = ica.plot_scores(eog_scores, show=False)  # Do not show the plot interactively
        ica_plot.savefig(os.path.join(subject_path, 'ica_component_scores.png'))  # Save the plot
        plt.close(ica_plot)  # Close the plot to free up memory
        
        a = abs(eog_scores).tolist()
        droping_components = 'one'
        if droping_components == 'one':  # find one maximum score
            ica.exclude = [a.index(max(a))]  # exclude the maximum index
        else:  # find two maximum scores
            a_2 = a.copy()
            a.sort(reverse=True)
            exclude_index = []
            for i in range(0, 2):
                for j in range(0, len(a_2)):
                    if a[i] == a_2[j]:
                        exclude_index.append(j)
            ica.exclude = exclude_index  # exclude these two maximum indices
        ica.apply(raw_data, verbose=False)  # apply the ICA
    except Exception as e:
        print(f"ICA error: {e}. Continuing without ICA.")
        
    # common average reference
    raw_data.set_eeg_reference('average', ch_type='eeg')  # , projection = True)
    filted_eeg_rawdata = np.array(raw_data.get_data())  # fetch the data from MNE.
    
    # Segment the data into 2-second windows (256 samples at 128Hz)
    segment_length = 256
    n_segments = filted_eeg_rawdata.shape[1] // segment_length
    segments = []
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = filted_eeg_rawdata[:, start_idx:end_idx]
        segments.append(segment)
    
    return segments

def signal_pro(input_data, subject_path):
    print("Trail preprocessing:")
    all_segments = []
    video_indices = []
    
    for i in trange(input_data.shape[0]):  # for each video sample
        segments = SignalPreProcess(input_data[i].copy(), subject_path)
        all_segments.extend(segments)
        # Keep track of which video each segment belongs to
        video_indices.extend([i] * len(segments))
    
    return all_segments, video_indices

def bandpower(input_data, band):
    band = np.asarray(band)
    low, high = band  # band is the tuple of (low, high)
    nperseg = (2 / low) * sf
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(input_data, sf, nperseg=nperseg)
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx_band])  # mean of the frequency bands

def emotion_label(labels, class_label):
    '''
    This function gives the valence/arousal and HVHA/HVLA/LAHV/LALV class labels
    '''
    em_labels = []
    if class_label == 'valence':
        for i in range(0, labels.shape[0]):
            if labels[i][0] > 5:  # high valence
                em_labels.append(1)
            else:  # low valence
                em_labels.append(0)
        return em_labels
    elif class_label == 'arousal':
        for i in range(0, labels.shape[0]):
            if labels[i][1] > 5:  # high arousal
                em_labels.append(1)
            else:  # low arousal
                em_labels.append(0)
        return em_labels
    elif class_label == 'all':
        for i in range(0, labels.shape[0]):
            if labels[i][0] > 5:  # high valence
                if labels[i][1] > 5:  # high arousal
                    em_labels.append(1)  # HVHA
                else:
                    em_labels.append(0)  # HVLA
            else:  # low valence
                if labels[i][1] > 5:  # high arousal
                    em_labels.append(2)  # LVHA
                else:  # low arousal
                    em_labels.append(3)  # LVLA
        return em_labels

def get_csv_file(subject, segments, video_indices, labels, datafiles_path):
    theta, alpha, beta, gamma = [], [], [], []
    theta_feat, alpha_feat, beta_feat, gamma_feat = [], [], [], []
    print("Feature Extraction:")
    
    for segment_idx in trange(len(segments)):
        segment = segments[segment_idx]
        for channel_no in range(len(segment)):
            channel_data = segment[channel_no]
            
            # PSD Features
            theta.append(bandpower(channel_data, theta_band_range))
            alpha.append(bandpower(channel_data, alpha_band_range))
            beta.append(bandpower(channel_data, beta_band_range))
            gamma.append(bandpower(channel_data, gamma_band_range))
    
    # PSD feature matrix
    total_segments = len(segments)
    theta = np.reshape(theta, (total_segments, 32))  # reshape to (n_segments, 32 channels)
    alpha = np.reshape(alpha, (total_segments, 32))
    beta = np.reshape(beta, (total_segments, 32))
    gamma = np.reshape(gamma, (total_segments, 32))
    
    # Add features name in the dataframe
    for i in range(0, len(channels)):
        theta_feat.append(channels[i] + '_theta')
        alpha_feat.append(channels[i] + '_alpha')
        gamma_feat.append(channels[i] + '_gamma')
        beta_feat.append(channels[i] + '_beta')
    
    df_theta = pd.DataFrame(theta, columns=theta_feat)
    df_alpha = pd.DataFrame(alpha, columns=alpha_feat)
    df_beta = pd.DataFrame(beta, columns=beta_feat)
    df_gamma = pd.DataFrame(gamma, columns=gamma_feat)
    
    # Make the subject name directory to save the csv file
    subject_path = os.path.join(datafiles_path, subject)
    os.makedirs(subject_path, exist_ok=True)  # Create directory if it doesn't exist
    
    # Map original video labels to segments
    valence_labels = emotion_label(labels, 'valence')
    arousal_labels = emotion_label(labels, 'arousal')
    four_class_labels = emotion_label(labels, 'all')
    
    segment_valence = [valence_labels[idx] for idx in video_indices]
    segment_arousal = [arousal_labels[idx] for idx in video_indices]
    segment_four_class = [four_class_labels[idx] for idx in video_indices]
    
    # Save the combined CSV file with all features and labels
    frames = [df_theta, df_alpha, df_beta, df_gamma]
    all_bands = pd.concat(frames, axis=1)  # join these 4 dataframes columns wise, rows are fixed
    all_bands['valence'] = segment_valence
    all_bands['arousal'] = segment_arousal
    all_bands['four_class'] = segment_four_class
    
    all_bands.to_csv(os.path.join(subject_path, subject + '_features.csv'), index=False, encoding='utf-8-sig')
    
    # Also save the individual class CSV files as in the original code
    all_bands_valence, all_bands_arousal, all_bands_all = all_bands.copy(), all_bands.copy(), all_bands.copy()
    all_bands_valence.to_csv(os.path.join(subject_path, subject + '_valence.csv'), index=False, encoding='utf-8-sig')
    all_bands_arousal.to_csv(os.path.join(subject_path, subject + '_arousal.csv'), index=False, encoding='utf-8-sig')
    all_bands_all.to_csv(os.path.join(subject_path, subject + '_four_class.csv'), index=False, encoding='utf-8-sig')

    # Calculate average power for each band
    avg_theta = np.mean(theta)
    avg_alpha = np.mean(alpha)
    avg_beta = np.mean(beta)
    avg_gamma = np.mean(gamma)

    # Plot bar graph
    bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
    avg_power = [avg_theta, avg_alpha, avg_beta, avg_gamma]

    plt.figure(figsize=(8, 6))
    plt.bar(bands, avg_power, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Frequency Bands')
    plt.ylabel('Average Power')
    plt.title(f'Average Power in Frequency Bands for Subject {subject}')
    plt.savefig(os.path.join(subject_path, f'{subject}_average_power.png'))  # Save the plot
    plt.close()
    
    print(f"Features saved to {subject_path}")
    print(f"Total segments: {total_segments}")
    print(f"Feature matrix shape: {theta.shape}")
    
    return all_bands

def main(args):
    # Construct the full path to the .dat file
    dat_file_path = os.path.join(args.deap_dataset_path, f"{args.subject}.dat")
    
    # Print the constructed path for debugging
    print(f"Looking for file: {dat_file_path}")
    
    # Check if the file exists
    if not os.path.exists(dat_file_path):
        print(f"Error: File not found at {dat_file_path}")
        sys.exit(1)
    
    # Load the .dat file
    try:
        with open(dat_file_path, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # raw_data has two keys 'data' and 'labels'
    data = raw_data['data']
    labels = raw_data['labels']
    
    # Exclude the first 3 seconds (384 samples) of baseline data
    reduced_eeg_data = data[0:40, 0:32, 384:8064]
    
    # Create subject directory
    subject_path = os.path.join(args.datafiles_path, args.subject)
    os.makedirs(subject_path, exist_ok=True)
    
    # Preprocess and segment the data
    segments, video_indices = signal_pro(reduced_eeg_data.copy(), subject_path)
    
    # Extract features and save CSV files
    features_df = get_csv_file(args.subject, segments, video_indices, labels, args.datafiles_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Global Variables
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    # SubBands range
    theta_band_range = (4, 8)   # drowsiness, emotional connection, intuition, creativity
    alpha_band_range = (8, 12)  # reflection, relaxation
    beta_band_range = (12, 30)  # concentration, problem solving, memory
    gamma_band_range = (30, 48) # cognition, perception, learning, multi-tasking

    sf = 128  # sampling frequency 128 Hz
    eeg_channels = np.array(channels)
    # Here 'all' refers for 'four class'
    class_labels = ['valence', 'arousal', 'all']

    parser.add_argument("--subject", type=str, default="s01", help="subject name")
    parser.add_argument("--deap_dataset_path", type=str, required=True, help="DEAP dataset path")
    parser.add_argument("--datafiles_path", type=str, required=True, help="Location of subject wise datafiles")
    args = parser.parse_args()
    main(args)