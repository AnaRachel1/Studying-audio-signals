import numpy as np
import librosa
from matplotlib import pyplot as plt
import librosa.display
import time

import libfmp.b
import libfmp.c3
import libfmp.c4



def SSM_regular (audio_path, plot = False,  n_fft = 4410, hop_length = 2205, filt_len = 40, 
                down_sampling = 10, L = 4, thresh_abs = 0.4, thresh_local = 0.15 ):
    
    ''' Take a audio, build a chromagram with some enhancements and return the SSM with Fs '''   
    
    def plot1_chroma( CHROMA, X_AXIS = 'frames'):
        plt.figure(figsize=(8, 2.5))
        librosa.display.specshow(CHROMA ,x_axis= X_AXIS, sr=Fs, hop_length= hop_length, cmap='gray_r')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    def plot_SSM_enhancements( SSM_original, S_smoothed, S_diagonal, S_thresh):
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
        libfmp.c4.subplot_matrix_colorbar(SSM_original, fig, ax[0], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'Original SSM')
        libfmp.c4.subplot_matrix_colorbar(S_smoothed, fig, ax[1], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Smoothed')
        libfmp.c4.subplot_matrix_colorbar(S_diagonal, fig, ax[2], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Diagonal')
        libfmp.c4.subplot_matrix_colorbar(S_thresh, fig, ax[3], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Thresh')

        plt.tight_layout()
        # plt.savefig(f'SSMs enhancements ({filt_len}, {down_sampling})')
        plt.show()
    
    
    # audio features
    x, Fs = librosa.load(audio_path)
    Fs = 22050
    Fs_C = Fs/2205 #10Hz
    
    #chromagram
    chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, 
                                        hop_length=hop_length, n_fft=n_fft)
    # chroma smoothed and normalized
    X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chroma, Fs_C, filt_len=filt_len,
                                                            down_sampling=down_sampling)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S = np.dot(np.transpose(X), X)  #similarity dot (inner)

    # diagonal filter
    tempo_rel_min = 1
    tempo_rel_max = 1
    num = 1 
    tempo_rel_set = libfmp.c4.compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num) 

    S_forward = libfmp.c4.filter_diag_mult_sm(S, L, direction=0)
    S_backward = libfmp.c4.filter_diag_mult_sm(S, L, direction=1)
    S_diagonal = np.maximum(S_forward, S_backward)


    # thresholding filter
    strategy1 = 'absolute'
    S_thresh = libfmp.c4.threshold_matrix(S_diagonal, thresh= thresh_abs, strategy=strategy1, scale=-2, penalty=0, binarize=0)
    strategy2 = 'local'
    thresh_loc = [thresh_local, thresh_local] 
    S_thresh = libfmp.c4.threshold_matrix(S_thresh, thresh=thresh_loc, strategy=strategy2, scale=-2, penalty=-2, binarize=0)

    if plot :
        # SSM original
        chroma_original = libfmp.c3.normalize_feature_sequence(chroma, norm='2', threshold=0.001)
        SSM_original = np.dot(np.transpose(chroma_original), chroma_original)  
        
        plot1_chroma(X, Fs)
        plot_SSM_enhancements(SSM_original, S, S_diagonal, S_thresh)
    
    return S_thresh, Fs_X
 

def SSM_energy (audio_path, energy = 1 , plot = False, n_fft = 4410, hop_length = 2205, filt_len = 40,
                down_sampling = 10, L = 4, thresh_abs = 0.4, thresh_local = 0.15 ):
    
    ''' Take a audio, build a chromagram with a energy layer and return the SSM with Fs '''   
  
    def plot1_chroma( CHROMA, X_AXIS = 'frames'):
        plt.figure(figsize=(8, 2.5))
        librosa.display.specshow(CHROMA ,x_axis= X_AXIS, sr=Fs, hop_length= hop_length, cmap='gray_r')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    def plot_SSM_enhancements( SSM_original, S_smoothed, S_diagonal, S_thresh):
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
        plt.title(" SSM of chroma modified with energy")
        libfmp.c4.subplot_matrix_colorbar(SSM_original, fig, ax[0], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'Original SSM')
        libfmp.c4.subplot_matrix_colorbar(S_smoothed, fig, ax[1], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Smoothed')
        libfmp.c4.subplot_matrix_colorbar(S_diagonal, fig, ax[2], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Diagonal')
        libfmp.c4.subplot_matrix_colorbar(S_thresh, fig, ax[3], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                                title=r'SSM Thresh')

        plt.tight_layout()
        plt.savefig(f'SSMs enhancements ({filt_len}, {down_sampling})')
        plt.show()
  
    # audio features
    x, Fs = librosa.load(audio_path)
    Fs = 22050
    Fs_C = Fs/2205 #10Hz

    # chromagram original
    chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, 
                                        hop_length=hop_length, n_fft=n_fft)
    # Itensity from spectogram
    S, phase = librosa.magphase(librosa.stft(x, hop_length= hop_length, n_fft= n_fft))
    rms = librosa.feature.rms(S=S, hop_length= hop_length, frame_length = n_fft)
    new_line = np.interp(rms, (rms.min(), rms.max()), (0.0, energy))

    chroma_modified= np.vstack((chroma, new_line))

    #filters
    X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chroma_modified, Fs_C, filt_len=filt_len,
                                                            down_sampling=down_sampling)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S_smoothed = np.dot(np.transpose(X), X)  

    # diagonal filter
    tempo_rel_min = 1
    tempo_rel_max = 1
    num = 1 
    tempo_rel_set = libfmp.c4.compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num) 

    S_forward = libfmp.c4.filter_diag_mult_sm(S_smoothed, L, direction=0)
    S_backward = libfmp.c4.filter_diag_mult_sm(S_smoothed, L, direction=1)
    S_diagonal = np.maximum(S_forward, S_backward)


    # thresholding filter
    strategy1 = 'absolute'
    S_thresh = libfmp.c4.threshold_matrix(S_diagonal, thresh=thresh_abs, strategy=strategy1, scale=-2, penalty=0, binarize=0)
    strategy2 = 'local'
    thresh2 = [thresh_local, thresh_local]
    S_thresh = libfmp.c4.threshold_matrix(S_thresh, thresh=thresh2, strategy=strategy2, scale=-2, penalty=-2, binarize=0)
    
    if plot :
        # SSM original
        chroma_original = libfmp.c3.normalize_feature_sequence(chroma, norm='2', threshold=0.001)
        SSM_original = np.dot(np.transpose(chroma_original), chroma_original)  
        
        plot1_chroma(X, Fs_X)
        plot_SSM_enhancements(SSM_original, S_smoothed, S_diagonal, S_thresh)
    
    return S_thresh, Fs_X

    
def find_family (alfa_start, alfa_end, SSM, Fs ):
    
    seg = [int(alfa_start*Fs), int(alfa_end*Fs)]

    S_seg = SSM[:,seg[0]:seg[1]+1] # sub Matrix corresponde a esse segmento
    D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg) # D = grafico dde pontos acumulados
    path_family = libfmp.c4.compute_optimal_path_family(D)

    N = SSM.shape[0] #  N (int): Length of feature sequence

    segment_family, coverage = libfmp.c4.compute_induced_segment_family_coverage(path_family)
    fitness, score, score_n, coverage, coverage_n, path_family_length = libfmp.c4.compute_fitness(
        path_family, score, N)

    # visualize family 
    #fn_ann = f'ann/{filename}_ann.csv'
    #ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)
    #fig, ax, im = libfmp.c4.plot_ssm_ann_optimal_path_family(SSM, ann_frames, seg, color_ann=color_ann, ylabel='Time (frames)')
    #plt.show()
    
    return segment_family, fitness


def main():
    path = 'music\Vultures.mp3'

    start1 = time.time()
    SSM1, Fs1  = SSM_regular(path,  )
    segment_family1, fitness1 = find_family(94, 120, SSM1, Fs1)
    end1 = time.time()
    
    print( "without energy")
    print (segment_family1) 
    print (" tempo gasto: ", round(end1 - start1))

if(__name__ == "__main__"):
    main()


