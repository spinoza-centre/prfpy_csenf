import numpy as np


class PRFStimulus2D(object):
    """PRFStimulus2D

    Minimal visual 2-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 screen_size_cm,
                 screen_distance_cm,
                 design_matrix,
                 TR,                 
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,
                 normalize_integral_dx=False,
                 **kwargs):
        """
        

        Parameters
        ----------
        screen_size_cm : float
            size of screen in centimeters. NOTE: prfpy uses a square design
            matrix. the screen_size_cm refers to the length of a side of this
            square in cm, i.e. for a rectangular screen,this is the length
            in cm of the smallest side.
        screen_distance_cm : float
            eye-screen distance in centimeters
        design_matrix : numpy.ndarray
            an N by t matrix, where N is [x, x]. 
            represents a square screen evolving over time (time is last dimension)
        TR : float
            Repetition time, in seconds
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.
        **kwargs : optional
            Use normalize_integral_dx = True to normalize the prf*stim sum as an integral.


        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.screen_size_cm = screen_size_cm
        self.screen_distance_cm = screen_distance_cm
        self.design_matrix = design_matrix
        if len(self.design_matrix.shape) >= 3 and self.design_matrix.shape[0] != self.design_matrix.shape[1]:
            raise ValueError  # need the screen to be square
        self.TR = TR
                
        # other useful stimulus properties
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict

        self.screen_size_degrees = 2.0 * \
            np.degrees(np.arctan(self.screen_size_cm /
                                 (2.0*self.screen_distance_cm)))

        oneD_grid = np.linspace(-self.screen_size_degrees/2,
                                self.screen_size_degrees/2,
                                self.design_matrix.shape[0],
                                endpoint=True)
        self.x_coordinates, self.y_coordinates = np.meshgrid(
            oneD_grid, oneD_grid)
        self.complex_coordinates = self.x_coordinates + self.y_coordinates * 1j
        self.ecc_coordinates = np.abs(self.complex_coordinates)
        self.polar_coordinates = np.angle(self.complex_coordinates)
        self.max_ecc = np.max(self.ecc_coordinates)

        # construct a standard mask based on standard deviation over time
        self.mask = np.std(design_matrix, axis=-1) != 0
        
        # whether or not to normalize the stimulus_through_prf as an integral, np.sum(prf*stim)*dx**2
        self.normalize_integral_dx = kwargs.pop('normalize_integral_dx', False)
        
        if self.normalize_integral_dx:
            self.dx = self.screen_size_degrees/self.design_matrix.shape[0]
        else:
            self.dx = 1


class PRFStimulus1D(object):
    """PRFStimulus1D

    Minimal visual 1-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 design_matrix,
                 mapping,
                 TR,
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,
                 **kwargs):
        """__init__


        Parameters
        ----------
        design_matrix : numpy.ndarray
            a 2D matrix (M by t). 
            represents inputs in an encoding space evolving over time (time is last dimension)
        mapping : numpy.ndarray, np.float
            for each of the columns in design_matrix, the value in the encoding dimension
            for example, in a numerosity experiment these would be the numerosity of presented stimuli
        TR : float
            Repetition time, in seconds
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.

        """
        self.design_matrix = design_matrix
        self.mapping = mapping
        self.TR = TR
        
        # other potentially useful stimulus properties
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict

        
class CFStimulus(object):
    
    """CFStimulus

    Minimal CF stimulus class. Creates a design matrix for CF models by taking the data within a sub-surface (e.g. V1).

    """
    
    
    def __init__(self,
                 data,
                 vertinds,
                 distances,**kwargs):
        
        """__init__


        Parameters
        ----------
        data : numpy.ndarray
            a 2D matrix that contains the whole brain data (second dimension must be time). 
            
            
         vertinds : numpy.ndarray
            a matrix of integers that define the whole-brain indices of the vertices in the source subsurface.
            
        distances : numpy.ndarray
            a matrix that contains the distances between each vertex in the source sub-surface.
            
        Returns
        -------
        data: Inherits data.
        subsurface_verts: Inherits vertinds.
        distance_matrix: Inherits distances.
        design_matrix: The data contained within the source subsurface (to be used as a design matrix).
        

        """
        self.data=data
        
        self.subsurface_verts=vertinds
        
        self.design_matrix=self.data[self.subsurface_verts,:]
        
        self.distance_matrix=distances


# ************************************************************************************************************
# CSenF functions
# (Contrast Sensitivity Function)
class CSenFStimulus(object):
    """CSenFStimulus

    CSenF stimulus creates a design matrix, used for fitting with CSenF model
    Actual stimuli presented to subjects are gratings of different spatial frequency and contrast levels
    We convert this into a binary design matrix of SF x Contrast x Time
    

    """

    def __init__(self,
                 SFs,
                 CONs,
                 SF_seq,
                 CON_seq,                
                 TR,
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,                 
                 ):
        """__init__


        Parameters
        ----------
        SFs : np.ndarray
            The set of unique SF values
        CONs : np.ndarray
            The set of unique contrast values
        SF_seq : np.ndarray (1 for each timepoint)
            Sequence of SF values in stimulus
        CON_seq : np.ndarray (1 for each timepoint)
            Sequence of contrast values in stimulus            
        TR : float
            Repetition time, in seconds
        
        Other info - mainly redundant, included for consistency with other prfpy components
        -------
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.
        """
        self.SFs = SFs
        self.log_SFs = np.log10(SFs)
        self.CONs = CONs
        self.CON_Ss = 100/CONs # contrast sensitivity
        self.SF_seq = SF_seq
        self.log_SF_seq = np.log10(SF_seq)        
        self.CON_seq = CON_seq
        self.CON_S_seq = 100/CON_seq 
        self.TR = TR
        
        self.n_SF = SFs.shape[0]
        self.n_CON = CONs.shape[0]
        self.n_TRs = SF_seq.shape[0]
        # Grids: true values
        self.SF_grid, self.CON_grid = np.meshgrid(self.SFs, self.CONs)        
        self.log_SF_grid, self.CON_S_grid = np.meshgrid(self.log_SFs, self.CON_Ss)
        
        # Grids: different SF and CON levels 
        # i.e., 0 empty, 1 = lowest SF, 2 = 2nd lowest SF, etc.
        # 0 empty, 1 = lowest CON, 2 = 2nd lowest CON, etc.
        SF_grid_id, CON_grid_id = np.meshgrid(np.arange(1,self.n_SF+1), np.arange(1,self.n_CON+1)) 
        # Round the values, so that we can match them to the rounded SF_seq, CON_seq (don't want to accidently miss a match, because not exact value)
        SFs_rnd = np.round(self.SFs, 3)     
        CONs_rnd = np.round(self.CONs, 3)   
        SF_seq_rnd = np.round(self.SF_seq, 3)
        CON_seq_rnd = np.round(self.CON_seq, 3)
        SF_seq_id = np.zeros_like(SF_seq_rnd, dtype=int)        
        for i,SF in enumerate(SFs_rnd):
            SF_seq_id[SF_seq_rnd==SF] = i+1
        CON_seq_id = np.zeros_like(CON_seq_rnd, dtype=int)
        for i,CON in enumerate(CONs_rnd):
            CON_seq_id[CON_seq_rnd==CON] = i+1


        # Create the design matrix (n SFS, n CON, n Timepoints)
        dm = np.zeros((self.n_CON, self.n_SF, self.n_TRs))
        for i in range(self.n_TRs):
            if CON_seq_id[i]!=0:
                this_frame = (SF_grid_id==SF_seq_id[i]) & (CON_grid_id==CON_seq_id[i])
                if this_frame.sum()==0:
                    print(self.SF_seq[i], self.CON_seq[i])
                    bloop
                dm[:,:,i] = np.copy(this_frame)
        
        self.design_matrix = dm

        # other useful stimulus properties [mainly obsolete]
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict

        # stimulus dx: again legacy
        self.dx = 1
