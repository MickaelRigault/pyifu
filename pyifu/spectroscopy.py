#! /usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import numpy as np
from propobject import BaseObject


__all__ = ["load_cube","load_spectrum",
            "get_spectrum", "get_slice", "get_cube"]

def load_cube(filename,**kwargs):
    """ Load a Cube from the given filename 
    
    Returns
    -------
    Cube
    """
    if not is_old_e3dformat(filename):
        return Cube(filename, **kwargs)
    return Cube.load_from_e3d(filename)

def get_cube(data, header=None, variance=None,
              lbda=None, spaxel_mapping=None,
                spaxel_vertices=None):
    """ 
    Parameters
    ----------
    spaxel_mapping: [dict]
        dictionary containing the coordinate of the spaxels with the given format:
        {id_:xy_ for id_,xy_ in zip(indexes,xy)}
        with xy_ the [x,y] coordinate of the id_-th index.
        (index could simply be range(number_of_spaxels) 

    """
    cube = Cube(None)
    cube.create(data, header=header, variance=variance,
                    lbda=lbda, spaxel_mapping=spaxel_mapping)
    if spaxel_vertices is not None:
        cube.set_spaxel_vertices(spaxel_vertices)
        
    return cube

def load_spectrum(filename,**kwargs):
    """ Load a Spectrum from the given filename 
    
    Returns
    -------
    Spectrum
    """
    return Spectrum(filename, **kwargs)

def load_slice(filename, **kwargs):
    """ """
    return Slice(filename, **kwargs)

def get_spectrum(lbda, flux, variance=None, header=None, logwave=None):
    """ Create a spectrum from the given data
    
    Parameters
    ----------
    lbda: [array]
        wavelength of the spectrum

    flux: [array]
        flux of the spectrum
    
    variance: [array/None] -optional-
        variance associated to the flux.

    header: [fits header / None]
        fits header assoiated to the data
        
    logwave: [None / bool] -optional-
        If the wavelength given in log of wavelength. 
        If you known set True (= given in log) or False (= given in angstrom)
        If let to None, this will test if the first wavelength is smaller or 
        higher than a default number (50).


    Returns
    -------
    Spectrum
    """
    spec = Spectrum(None)
    spec.create(data=flux, variance=variance, header=header, lbda=lbda, logwave=logwave)
    return spec

def get_slice(data, xy, spaxel_vertices=None,
                variance=None, indexes=None, lbda=None,
                header=None):
    """ 
    Parameters
    ----------

    """
    slice_ = Slice(None)
    if indexes is None:
        indexes = np.arange(len(xy))
    if len(data) != len(xy):
        raise ValueError("xy and data do not have the same lenth %d vs. %d"%(len(xy), len(data)))
    
    spaxel_mapping = {id_:xy_ for id_,xy_ in zip(indexes,xy)}

    slice_.create(data, variance=variance, lbda=lbda,spaxel_mapping=spaxel_mapping)
    if spaxel_vertices is not None:
        slice_.set_spaxel_vertices(spaxel_vertices)
    if header is not None:
        slice_.set_header(header)
    return slice_


def synthesize_photometry3d(lbda, flux, filter_lbda, filter_trans):
    """ """
    from .tools import nantrapz
    filter_interp = np.interp(lbda, filter_lbda, filter_trans)
    
    dphotons = (filter_interp[:,None] * flux) * lbda[:,None] * 5.006909561e7
    refnorm  = flux/flux # nan stay nan, rest is 1
    norm_    = (filter_interp[:,None] *  refnorm)* lbda[:,None] * 5.006909561e7
    
    return nantrapz(dphotons.T,lbda)/nantrapz(norm_.T, lbda)

                

def synthesize_photometry(lbda, flux, filter_lbda, filter_trans,
                          normed=True):
    """ Get Photometry from the given spectral information through the given filter.

    This function converts the flux into photons since the transmission provides the
    fraction of photons that goes though.


    Parameters
    -----------
    lbda, flux: [array]
        Wavelength and flux of the spectrum from which you want to synthetize photometry
        
    filter_lbda, filter_trans: [array]
        Wavelength and transmission of the filter.

    normed: [bool] -optional-
        Shall the fitler transmission be normalized?

    Returns
    -------
    Float (photometric point)
    """
    # ---------
    # The Tool
    from .tools import nantrapz
    def integrate_photons(lbda, flux, step, flbda, fthroughput):
        """ """
        filter_interp = np.interp(lbda, flbda, fthroughput)
        dphotons = (filter_interp * flux) * lbda * 5.006909561e7
        return nantrapz(dphotons,lbda) if step is None else np.sum(dphotons*step)
    
    # ---------
    # The Code
    normband = 1. if not normed else \
      integrate_photons(lbda, np.ones(len(lbda)),None,filter_lbda,filter_trans)
      
    return integrate_photons(lbda,flux,None,filter_lbda,filter_trans)/normband

# ----------- #
# Redshifting #
# ----------- #
def deredshift(x, y, z, variance=None, exp=3):
    """Deredshift spectrum from z to 0, and apply a (1+z)**exp
    flux-correction.

    exp=3 is for erg/s/cm2/A spectra to be latter corrected using proper
    (comoving) distance but *not* luminosity distance.

    Parameters
    ----------
    x,y : [array]
        wavelength and flux or the spectrum. 
        y is expected in erg/s/cm2/A (so x in A)

    variance: [array] -optional-
        spectral variance if any (square of flux units.)

    exp: [float] -optional-
        exposant for the redshift flux dilution 
        [exp=3 is for erg/s/cm2/A spectra]

    Returns
    -------
    arrays (x, y [and variance if any] corresponding to the deredshifted spectrum)

    """
    zp1 = 1 + z
    x_rest = x/zp1           # Wavelength correction
    y_rest = y*(zp1**exp)      # Flux correction
    if variance is not None:
        variance_rest = variance*(zp1**exp)**2

    return (x_rest, y_rest , variance_rest) if variance is not None else (x_rest, y_rest)




# Not in .tools.py to avoid loading matplotlib.
def is_arraylike(a):
    """ Tests if 'a' is an array / list / tuple """
    return isinstance(a, (list, tuple, np.ndarray) )

def is_old_e3dformat(filename):
    """ """
    from astropy.io.fits import getheader
    header = getheader(filename)
    return "EURO3D" in header and header["EURO3D"] in [True, "T",'true', "True"]

##########################
#                        #
# Low-Level SpecSource   #
#                        #
##########################
class SpecSource( BaseObject ):
    """ Virtual Object that contains the similaties between Spectrum and Cube. """
    
    PROPERTIES         = ["rawdata","rawvariance","lbda","header"]
    SIDE_PROPERTIES    = ["filename","fits","header"]
    DERIVED_PROPERTIES = ["data","variance","spec_prop"]
    
    def __init__(self,filename,
                 dataindex=0,
                 headerindex=None):
        """ Initializes the object
        
        Parameters
        ----------
        filename: [string/None]  
            The name of the fits file containing the spectroscopic object. 
            You can set None to create an incomplet object. 
            Run create later on to have access to its full functionality.

        dataindex: [int]         
           Index of the hdu-table where the data are
           registered. If you do not know leave 0, 
           for some cases (like MUSE) uses 1.

        headerindex: [int/None] 
            Index of the hdu-table where the header you want to use it.
            The default header is the one contained in the dataindex 
            hdu table.

        Returns
        -------
        Void
        """
        self.__build__()
        if filename is not None:
            self.load(filename,dataindex=dataindex,
                      headerindex=headerindex)

    def __build__(self):
        """ Low Level method that builds the _build_properties
        It for instance contains the information about Header keywords
        """
        super(SpecSource, self).__build__()
        self._build_properties= dict(
                stepkey   = "CDELT",
                startkey  = "CRVAL",
                lengthkey = "NAXIS")
        
    # =================== #
    #   Main Metods       #
    # =================== #
    # -------- #
    #  I/O     #
    # -------- #

    def writeto(self,savefile,force=True,saveerror=False):
        raise NotImplementedError("Write method not implemented for SpecSources")

    def load(self,  filename, dataindex=0, varianceindex=1, headerindex=None):
        raise NotImplementedError("Load method not implemented for SpecSources")

    def scale_by(self, coef, onraw=True):
        """ divide the data by the given scaling factor 
        If this object has a variance attached, the variance will be divided by the square of `coef`.
        Parameters
        ----------
        coef: [float or array of]
            scaling factor for the data 

        onraw: [bool] -optional-
            scaley from rawdata or from data (that could already be scaled)

        Returns
        -------
        Void, affect the object (data, variance)
        """
        if not is_arraylike(coef) or len(coef)==1 or len(coef) == len(self.data):
            
            self._derived_properties["data"]  = self.rawdata / coef if onraw else self.data / coef
            if self.has_variance():
                self._derived_properties["variance"]  = self.rawvariance / coef**2 if onraw else self.variance / coef**2
        else:
            raise ValueError("scale_by is not able to parse the shape of coef.", np.shape(coef))
    # -------- #
    #  SETTER  #
    # -------- #
    def create(self, data, header=None,
                    variance=None,
                    lbda=None, logwave=None):
        """  High level setting method.

        Parameters
        ----------
        data: [array]
            The array containing the data
            
        variance: [array] -optional-
            The variance associated to the data. 
            This must have the same shape as data
           
        header: [pyfits/astropy fits header / None]
            Header associated to the fits file. It could contains
            the lbda information (step, size, start). This is needed
            if lbda is not given.

        lbda: [array] -optional-
            Provide the wavelength array associated with the data.
            This is not mendatory if the header contains this information
            (step, size and start values). 
            N.B: You can always use set_lbda() later on.

        logwave: [None / bool] -optional-
            If the wavelength given in log of wavelength. 
            If you known set True (= given in log) or False (= given in angstrom)
            If let to None, this will test if the first wavelength is smaller or 
            higher than a default number (50).

        Returns
        -------
        Void
        """
        self.set_header( header )
        self.set_data(data, variance, lbda, logwave=logwave)

    def set_data(self, data, variance=None, lbda=None, logwave=None):
        """ Set the spectral data 

        Parameters
        ----------
        data: [array]
            The array containing the data
        
        variance: [array] -optional-
            The variance associated to the data. 
            This must have the same shape as data
           
        lbda: [array] -optional-
            Provide the wavelength array associated with the data.
            This is not mendatory if the header contains this information
            (step, size and start values). 
            N.B: You can always use set_lbda() later on.
            
        logwave: [None / bool] -optional-
            If the wavelength given in log of wavelength. 
            If you known set True (= given in log) or False (= given in angstrom)
            If let to None, this will test if the first wavelength is smaller or 
            higher than a default number (50).

        Returns
        -------
        Void
        """
        # - 3d data
        self._properties["rawdata"] = np.asarray(data)
        # - Variance
        if variance is not None:
            if np.shape(variance) != np.shape(data):
                raise TypeError("variance and data do not have the same shape")
            self._properties["rawvariance"] = np.asarray(variance)
            
        # - Wavelength
        if lbda is not None:
            self.set_lbda(lbda, logwave=logwave)
        else:
            try:
                self.load_lbda_fromheader()
            except:
                warnings.warn("No lbda given and cannot load it from the header")
        
            
    def set_lbda(self, lbda, logwave=None, setheader=False):
        """ Set the wavelength associated with the data.
        
        Parameters
        ----------
        lbda: [None / array]
            The wavelength array associated to the data. 
            lbda can have several shape:

            - None: This reset the lbda entry. 
              If so, if called lbda will be extracted from the header 
              if possible. It would remain None otherwise.
                    
            - Array of constant step: the favored structure.
              If the step of the given array is constant (e.g. np.linspace array)
              then this array will be decomposed into step, size and start
              which value will feed the header. The self.lbda is then built from the 
              header.
              This then allow easy saving of the data following the 3d format.
            
            - Array with non constant step: This array will simply 
              be saved as is in self.lbda. 
                  
        logwave: [None / bool] -optional-
            If the wavelength given in log of wavelength. 
            If you known set True (= given in log) or False (= given in angstrom)
            If let to None, this will test if the first wavelength is smaller or 
            higher than a default number (50).

        Returns
        -------
        Void
        """
        if lbda is None:
            self._properties["lbda"] = None
            return
        if len(np.atleast_1d(lbda)) == 1:
            self._properties["lbda"] = lbda
            return
        
        # - unique step array?
        if len(np.unique(np.round(lbda[1:]-lbda[:-1], decimals=4)))==1 and setheader:
            # slower and a bit convoluted but ensure class' consistency
            self._lbda_to_header_(lbda, logwave)
            self.load_lbda_fromheader()
        else:
            self._properties["lbda"] = np.asarray(lbda)

        
    def set_header(self, header):
        """ Attach a header. 
        If the given header is None, an empty header will be attached.
        """
        self._side_properties["header"] = header
        self._header_to_spec_prop_()

    def set_filename(self, filename):
        """ """
        self._side_properties["filename"] = filename
        
    # =================== #
    #   Properties        #
    # =================== #
    # --------- #
    #  Basics   #
    # --------- #
    @property
    def lbda(self):
        """ """
        if self._properties["lbda"] is None:
            self.load_lbda_fromheader()
        return self._properties["lbda"]

    def load_lbda_fromheader(self):
        """ """
        self._properties["lbda"] = self._lbda_from_header_()
        
    @property
    def rawdata(self):
        """ """
        return self._properties["rawdata"]

    def has_data(self):
        """ """
        return self.rawdata is not None

    @property
    def rawvariance(self):
        """ """
        return self._properties["rawvariance"]
    
    @property
    def variance(self):
        """ """
        if self._derived_properties["variance"] is None:
            return self.rawvariance
        return self._derived_properties["variance"]
    
    def has_variance(self):
        """ Tests if a variance has been set. True means yes. """
        return self.rawvariance is not None

    # - derived
    @property
    def data(self):
        """ """
        if self._derived_properties["data"] is None:
            return self.rawdata
        return self._derived_properties["data"]

    @property
    def spec_prop(self):
        if self._derived_properties["spec_prop"] is None:
            self._derived_properties["spec_prop"] = {}
        return self._derived_properties["spec_prop"]

    def _has_spec_setup_(self):
        """ Test if the spaxels properties has been step up"""
        return len(self.spec_prop.keys())>0

    # - Side prop
    @property
    def filename(self):
        """ The filename of the spectral data (if any) """
        return self._side_properties["filename"]
    
    @property
    def fits(self):
        """ Fits file opened by astropy.io.fits (if any) """
        return self._side_properties["fits"]
    
    # ---------- #
    #  Header    #
    # ---------- #
    @property
    def header(self):
        """ """
        if self._side_properties['header'] is None:
            from astropy.io.fits import Header
            self._side_properties["header"] = Header()
        return self._side_properties["header"]

    def _lbda_from_header_(self):
        """ """
        if not self._has_spec_setup_():
            return None

        if self.spec_prop["logwave"] is None:
            self.spec_prop["logwave"] = bool( self.spec_prop["lstart"] < 50 )
            
        return np.arange(self.spec_prop["lspix"]) * self.spec_prop["lstep"] + self.spec_prop["lstart"] \
          if not self.spec_prop["logwave"] else np.exp(np.arange(self.spec_prop["lspix"]) * self.spec_prop["lstep"] + self.spec_prop["lstart"])

    def _lbda_to_header_(self, lbda, logwave=None):
        """ converts the lbda array into step, size and start and feed it to
        spec_prop

        Returns
        -------
        Void
        """
        self.spec_prop["lstep"]  = np.unique(lbda[1:]-lbda[:-1])[0]
        self.spec_prop["lspix"]  = len(lbda)
        self.spec_prop["lstart"] = lbda[0]
        self.spec_prop["logwave"] = bool(self.spec_prop["lstart"] < 50) if logwave is None or logwave=="None" else bool(logwave)
        
        self.header.set('%s1'%self._build_properties["lengthkey"],self.spec_prop["lspix"])   
        self.header.set('%s1'%self._build_properties["stepkey"],self.spec_prop["lstep"])
        self.header.set('%s1'%self._build_properties["startkey"],self.spec_prop["lstart"])
    # -----------
    # - internal    
    @property
    def _l_spix(self):
        if "lspix" not in self.spec_prop or self.spec_prop["lspix"] is None:
            self.spec_prop["lspix"] = np.shape(self.data)[0]
        return self.spec_prop["lspix"]

##############################
#                            #
#   Spectrum                 #
#                            #
##############################
class Spectrum( SpecSource ):
    """ """
    # ================ #
    #  Main Method     #
    # ================ #
    def _build_hdulist_(self, saveerror=False):
        """ The fits hdulist that should be saved.

        Parameters
        ----------
        saveerror:  [bool]      
            Set this to True if you wish to record the error and not the variance
            in you first hdu-table. if False, the table will be called
            VARIANCE and have self.v; if True, the table will be called
            ERROR and have sqrt(self.v)

        Returns
        -------
        Void
        """
        from astropy.io.fits import PrimaryHDU, ImageHDU
        hdul = []
        # -- Data saving
        hdul.append(PrimaryHDU(self.data, self.header))
        
        # -- Variance saving
        if self.has_variance():
            hduVar  = ImageHDU(np.sqrt(self.variance), name='ERROR') if saveerror else\
              ImageHDU(self.variance, name='VARIANCE')
            hdul.append(hduVar)
            
        if self._has_spec_setup_():
            if self.has_variance():
                hduVar.header.set('%s1'%self._build_properties["lengthkey"],self.spec_prop["lspix"])   
                hduVar.header.set('%s1'%self._build_properties["stepkey"],self.spec_prop["lstep"])
                hduVar.header.set('%s1'%self._build_properties["startkey"],self.spec_prop["lstart"])
        else:
            hdul.append(ImageHDU(self.lbda, name='LBDA'))
                
        return hdul
        
    def writeto(self, savefile, force=True, saveerror=False, ascii=False):
        """ Save the Spectrum into the given `savefile`

        Parameters
        ----------
        savefile: [string]      
            Fullpath of the file where the spectrum will be saved.

        force: [bool] -optional-
            If the file already exist, shall this overwrite it ? 
            (hence erasing the former one)

        saveerror:  [bool]      
            Set this to True if you wish to record the error and not the variance
            in you first hdu-table. if False, the table will be called
            VARIANCE and have self.v; if True, the table will be called
            ERROR and have sqrt(self.v)

        ascii: [bool]
            Save the current object as ascii data table:
            - Starts with the header: #KEYWORD VALUE 
            - Follows with the data: LBDA FLUX ERROR/VARIANCE [see saveerror]

        Returns
        -------
        Void
        """
        if not ascii:
            from astropy.io.fits import HDUList
            hdulist = HDUList(self._build_hdulist_(saveerror))
            hdulist.writeto(savefile,overwrite=force)
        else:
            fileout = open(savefile.replace(".fits", ".txt"),"w")
            for k,v in self.header.items():
                fileout.write("# %s: %s\n"%(k.upper(),v))
            varerr_ = [np.nan] * len(self.lbda) if not self.has_variance() else self.variance if not saveerror else np.sqrt(self.variance)
            
            for l_,f_,v_ in zip(self.lbda, self.data, varerr_):
                fileout.write("%.1f %.3e %.3e\n"%(l_,f_,v_))
            fileout.close()

    def load(self, filename, dataindex=0, varianceindex=1, headerindex=None):
        """ 

        lbda - If an hdu column of the fits file is name:
               "LBDA" or "LAMBDA" or "WAVE" or "WAVELENGTH" or "WAVELENGTHS",
               the column will the used as lbda
        
        """
        from astropy.io.fits import open as fitsopen
        self._side_properties["filename"] = filename
        self._side_properties["fits"]     = fitsopen(filename)
        
        if headerindex is None:
            headerindex = dataindex
            
        # Get the data 
        data = self.fits[dataindex].data
        
        # Get the variance or the error
        if varianceindex is not None and len(self.fits)>varianceindex:
            if self.fits[varianceindex].name.upper() in ["ERR","ERROR", "ERRORS"]:
                variance = self.fits[varianceindex].data**2
            else:
                variance = self.fits[varianceindex].data
        else:
            variance = None
            
        # Get the LBDA if any
        lbda_ = [f.data for f in self.fits if f.name.upper() in ["LBDA","LAMBDA", "WAVE", "WAVELENGTH","WAVELENGTHS"]]
        lbda = None if len(lbda_)==0 else lbda_[0]

        # --- Create the object
        self.create(data=data, header=self.fits[headerindex].header,
                    variance=variance, lbda=lbda)

    # --------- #
    #  Tools    #
    # --------- #
    def synthesize_photometry(self, filter_lbda, filter_trans, on="data"):
        """ Measure the photometry at wich one would have observed this spectra
        using the given filter.
        
        This method uses 'synthesize_photometry', which converts the flux into photons
        since the transmission provides the fraction of photons that goes though.


        Parameters
        -----------
        
        filter_lbda, filter_trans: [array]
            Wavelength and transmission of the filter.
            
        normed: [bool] -optional-
            Shall the fitler transmission be normalized?

        Returns
        -------
        Float (photometric point), Float/None (variance, only if this has a variance and on in ['data','rawdata'])
        """
        return synthesize_photometry(self.lbda, eval("self.%s"%on),
                                         filter_lbda, filter_trans),\
               synthesize_photometry(self.lbda, self.variance,
                                         filter_lbda, filter_trans) if self.has_variance() \
                                         and on in ["data","rawdata"] else None

    def filter(self, new_disp):
        """ apply a Gaussian filtering to the current spectrum and returns a copy of the filtered one 

        Parameters
        ----------
        new_disp: [float]
            scale (sigma) of the filtering in pixel (lbda) unit.
            
        Returns 
        -------
        Spectrum
        """
        from scipy.ndimage.filters import gaussian_filter
        data_ = gaussian_filter(self.data, new_disp)
        var_  = gaussian_filter(self.variance, new_disp) if self.has_variance() else None
        spec_ = self.copy()
        spec_._derived_properties["data"] = None
        spec_.create(data_, lbda=self.lbda, variance=var_, header=self.header.copy())
        return spec_
        
    def reshape(self, new_lbda, kind="linear", fill_value="extrapolate", **kwargs):
        """ Create a copy of the current spectrum with a new wavelength shape.
        Flux and variances (if any) will be reshaped accordingly
    
        This uses scipy.interpolate.interp1d

        Parameters
        ----------
        new_lbda: [array]
            new wavelength array (in Angstrom)

        // Scipy interpolate options

        kind : [str or int] -optional-
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        
        fill_value: [array-like or (array-like, array_like) or "extrapolate"] -optional-
            - if a ndarray (or float), this value will be used to fill in for
            requested points outside of the data range. If not provided, then
            the default is NaN. The array-like must broadcast properly to the
            dimensions of the non-interpolation axes.
            - If a two-element tuple, then the first element is used as a
            fill value for ``x_new < x[0]`` and the second element is used for
            ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
            list or ndarray, regardless of shape) is taken to be a single
            array-like argument meant to be used for both bounds as
            ``below, above = fill_value, fill_value``.

        **kwargs goes to scipy.interpolate.interp1d

        Returns
        -------
        Spectrum
        """
        from scipy.interpolate import interp1d
        data_ = interp1d(self.lbda, self.data,     kind=kind, fill_value="extrapolate", **kwargs)(new_lbda)
        var_  = interp1d(self.lbda, self.variance, kind=kind, fill_value="extrapolate", **kwargs)(new_lbda) \
          if self.has_variance() else None
        spec_ = self.copy()
        spec_._derived_properties["data"] = None
        spec_.create(data_, lbda=new_lbda, variance=var_, header=self.header.copy())
        return spec_
        
    # --------- #
    #  PLOTTER  #
    # --------- #
    def show(self, toshow="data", ax=None, savefile=None, show=True,
                 show_zero=False,**kwargs):
        """ Display the spectrum.
        
        Parameters
        ----------
        toshow: [string] -optional-
            Variable you want to display. anything accessible as self.`toshow` that 
            has the same size as the wavelength. 
            If toshow is data or rawdata, the variance will automatically be added
            if it exists.
            Do not change this is you have a doubt.

        ax: [matplotlib.Axes] -optional-
            Provide the axes where the spectrum will be drawn.
            If None this will create a new one inside a new figure
            
        savefile: [string/None] -optional-
            Would you like to save the data? If so give the name of this
            file where the plot will be saved.
            You can provide an extention (.pdf or .png) if you don't both the
            .pdf and .png will be created.

        show: [bool] -optional-
            If you do not save the data (see savefile), shall the plot be shown?

        **kwargs goes to specplot (any matplotlib axes.plot entry will work)

        Returns
        -------
        Void
        """
        from .tools import figout, specplot, mpl
        # - Axis definition
        if ax is None:
            fig = mpl.figure(figsize=[6.5,3.5])
            ax = fig.add_axes([0.15,0.15,0.8,0.75])
            ax.set_xlabel(r"Wavelength", fontsize="large")
            ax.set_ylabel(r"Flux", fontsize="large")
        else:
            fig  = ax.figure

        
        # - Ploting
        spec = eval("self.%s"%toshow)
        var  = self.variance if toshow in ["data", "rawdata"] and self.has_variance() else None
        if show_zero:
            ax.axhline(0, ls="-", color="0.7", lw=1)
        pl = ax.specplot(self.lbda, spec, var=var, **kwargs)
        
        # - out
        fig.figout(savefile=savefile, show=show)
        
        return {"plot":pl, "ax":ax, "fig":fig}
    # ================ #
    #  Properties      #
    # ================ #
    def _header_to_spec_prop_(self):
        """ """
        self.spec_prop["lspix"]  = self.header.get('%s1'%self._build_properties["lengthkey"])
        self.spec_prop["lstep"]  = self.header.get('%s1'%self._build_properties["stepkey"])
        self.spec_prop["lstart"] = self.header.get('%s1'%self._build_properties["startkey"])
        self.spec_prop["logwave"]= self.header.get('logwave', None)

##############################
#                            #
#   Cubes                    #
#                            #
##############################
class SpaxelHandler( SpecSource ):
    """ """
    PROPERTIES = ["spaxel_mapping","spaxel_vertices"]

    @classmethod
    def from_data(cls, data, variance=None, header=None, lbda=None,
                      spaxel_mapping=None, spaxel_vertices=None):
        """ """
        this = cls(None)
        this.create(data, header=header, variance=variance,
                    lbda=lbda, spaxel_mapping=spaxel_mapping)
        if spaxel_vertices is not None:
            this.set_spaxel_vertices(spaxel_vertices)
        
        return this

    def get_new(self, newdata=None, newvariance=None,
                    newheader=None, newlbda=None,
                    newspaxel_mapping=None, newspaxel_vertices=None
                    ):
        if newdata is None:
            newdata = self.data
        if newvariance is None:
            newvariance = self.variance
        elif type(newvariance) is str and newvariance == "None":
            newvariance = None
            
        if newheader is None:
            newheader = self.header
        if newlbda is None:
            newlbda = self.lbda

        if newspaxel_mapping is None:
            newspaxel_mapping = self.spaxel_mapping
            
        if newspaxel_vertices is None:
            newspaxel_vertices = self.spaxel_vertices

        return self.__class__.from_data(data=newdata, variance=newvariance,
                                        header=newheader, lbda=newlbda,
                                        spaxel_mapping=newspaxel_mapping, spaxel_vertices=newspaxel_vertices)
    # ================ #
    #  Main Method     #
    # ================ #
    def create(self, data ,header=None,
                    variance=None,
                    lbda=None, spaxel_mapping=None, **kwargs):
        """  High level setting method.

        Parameters
        ----------
        data: [array]
            The array containing the data
            
        variance: [array] -optional-
            The variance associated to the data. 
            This must have the same shape as data
           
        header: [pyfits/astropy fits header / None]
            Header associated to the fits file. It could contains
            the lbda information (step, size, start). This is needed
            if lbda is not given.

        lbda: [array] -optional-
            Provide the wavelength array associated with the data.
            This is not mendatory if the header contains this information
            (step, size and start values). 
            N.B: You can always use set_lbda() later on.

        spaxel_mapping: [array] -optional-
            Provide how the data are organised in the 2D grid. 
            If the given data are 3D cubes (lambda,x,y) no need
            to provide the mapping, this will be made automatically
            (see `set_data`)

        Returns
        -------
        Void
        """
        self.set_header(header)
        self.set_data(data, variance, lbda,
                      spaxel_mapping=spaxel_mapping, **kwargs)

    def set_data(self, data, variance=None, lbda=None, spaxel_mapping=None,
                     logwave=None, lbdatoheader=False):
        """ Set the spectral data 

        Parameters
        ----------
        data: [array]
            The array containing the data
        
        variance: [array] -optional-
            The variance associated to the data. 
            This must have the same shape as data
           
        lbda: [array] -optional-
            Provide the wavelength array associated with the data.
            This is not mendatory if the header contains this information
            (step, size and start values). 
            N.B: You can always use set_lbda() later on.
            
        spaxel_mapping: [array] -optional-
            Provide how the data are organised in the 2D grid. 
            If the given data are 3D cubes (lambda,x,y) no need
            to provide the mapping, this will be made automatically

        Returns
        -------
        Void
        """
        if spaxel_mapping is None and hasattr(self, "spaxel_mapping") and self.spaxel_mapping is not None and len(self.spaxel_mapping)>0:
            if len(data) != self.nspaxels or (variance is not None and len(variance) != self.nspaxels):
                raise ValueError("size of data and/or variance (if any) does not match that of existing spaxel_mapping")
            
        # - 3d data
        self._properties["rawdata"] = np.asarray(data)
        # - Variance
        if variance is not None:
            if np.shape(variance) != np.shape(data):
                raise TypeError("variance and data do not have the same shape")
            self._properties["rawvariance"] = np.asarray(variance)
            self._derived_properties["variance"] = None
            
        if spaxel_mapping is not None or not hasattr(self, "spaxel_mapping"):
            self.set_spaxel_mapping(spaxel_mapping)
        
        # - Wavelength
        if lbda is not None:
            self.set_lbda(lbda, logwave=logwave, setheader=lbdatoheader)
        else:
            try:
                self.load_lbda_fromheader()
            except:
                warnings.warn("No lbda given and cannot load it from the header")

    def set_spaxel_mapping(self, spaxel_mapping):
        """ Provide how the data are organised in the 2D grid. 
        If the data corresponds to a 3D cubes (lambda,x,y) 
        you can set `spaxel_mapping = None` and this will be made it 
        automatically

        Parameters
        ----------
        spaxel_mapping: [list/None]
            if provided this list must be a dictionary with this format:
            {index_0: [x_0, y_0],
             index_1: [x_1, y_1], 
             etc.
            }

        Returns
        -------
        Void
        """
        if spaxel_mapping is None:
            # - Is that a 3D cube?
            if self.is_3d_cube():
                spaxel_mapping = {}
                [[spaxel_mapping.setdefault(i+self.data.shape[1]*j,[i,j])
                      for i in range(self.data.shape[1])] for j in range(self.data.shape[2])]
                self.set_spaxel_mapping(spaxel_mapping)
            else:
                raise ValueError("The cube is not a 3D cube, you must provide a spaxel_mapping")
            
        elif type(spaxel_mapping) is not dict:
            raise TypeError("The given `spaxel_mapping` must be a dictionary. See doc")

        # - make sure the format is correct
        self._properties["spaxel_mapping"] = {i:np.asarray(v).tolist() for i,v in spaxel_mapping.items()}

    def set_spaxel_vertices(self, xy):
        """ Provide the countours of the polygon that define the spaxel.
        This will be the default spaxel structure so the central value of
        the given vertices should be (0,0).
        
        Examples:
        ---------
        for a square:   xy = [[-0.5, 0.5],[0.5, 0.5],[-0.5,-0.5],[0.5,-0.5]] 
        for a hexagone: xy = [[1.,0], [1/2., np.sqrt(3/2.)], [-1/2., np.sqrt(3/2.)], [-1,0], 
                             [-1/2., -np.sqrt(3/2.)], [1/2., -np.sqrt(3/2.)]]

        """
        if np.shape(xy)[1] != 2:
            raise TypeError("The given vertices must be a (n,2) array")
        
        self._properties["spaxel_vertices"] = np.asarray(xy)
        
    # -------- #
    #   I/O    #
    # -------- #
    def writeto(self, savefile, format=None, **kwargs):
        """ """
        if format is None:
            format = savefile.split(".")[-1]

        if format == "fits":
            self.to_fits(savefile, **kwargs)
        elif format in ["hdf","hdf5","h5"]:
            self.to_hdf(savefile, **kwargs)
        else:
            raise NotImplementedError(f"only fits and hdf5 format implemented, {format} given")

    def to_hdf(self, savefile, keys={}):
        """ """
        pandas_dict = self.to_pandas()
        for key, d in pandas_dict.items():
            if d is not None:
                d.to_hdf(savefile, key=key)

    def to_fits(self, savefile, force=True, saveerror=False, headerbased=True, lbdainheader=True):
        """ Save the cube the given `savefile`

        Parameters
        ----------
        savefile: [string]      
            Fullpath of the file where the cube will be saved.

        force: [bool] -optional-
            If the file already exist, shall this overwrite it ? 
            (hence erasing the former one)

        saveerror:  [bool] -optional- 
            Set this to True if you wish to record the error and not the variance
            in you first hdu-table. if False, the table will be called
            VARIANCE and have self.v; if True, the table will be called
            ERROR and have sqrt(self.v)

        Returns
        -------
        Void
        """
        from astropy.io.fits import PrimaryHDU, ImageHDU, HDUList
        hdul = []
        # -- Data saving
        hduP = PrimaryHDU(self.data, header=self.header)
        hdul.append(hduP)
    
        if self.has_variance():
            hduVar  = ImageHDU(np.sqrt(self.variance), name='ERROR') if saveerror else\
                      ImageHDU(self.variance, name='VARIANCE') 
            hdul.append(hduVar)

        
        naxis = 3 if self.is_3d_cube() else len(np.shape(self.data))
        
        if headerbased:
            if "lstep" in self.spec_prop.keys():
                hduP.header.set('NAXIS',naxis)
                hduP.header.set('%s%d'%(self._build_properties["lengthkey"],naxis), self.spec_prop["lspix"])   
                hduP.header.set('%s%d'%(self._build_properties["stepkey"],naxis),   self.spec_prop["lstep"])
                hduP.header.set('%s%d'%(self._build_properties["startkey"],naxis),  self.spec_prop["lstart"])
            elif self.lbda is not None:
                hdul.append(ImageHDU(self.lbda, name='LBDA'))
    
            if self.is_3d_cube():
                hduP.header.set('NAXIS',naxis)
                hduP.header.set('%s1'%self._build_properties["lengthkey"],self.spec_prop["wspix"])
                hduP.header.set('%s2'%self._build_properties["lengthkey"],self.spec_prop["nspix"])  
                hduP.header.set('%s1'%self._build_properties["stepkey"],self.spec_prop["wstep"])
                hduP.header.set('%s2'%self._build_properties["stepkey"],self.spec_prop["nstep"])
                hduP.header.set('%s1'%self._build_properties["startkey"],self.spec_prop["wstart"])
                hduP.header.set('%s2'%self._build_properties["startkey"],self.spec_prop["nstart"])
            elif naxis==2 and lbdainheader:
                hduP.header.set('%s1'%self._build_properties["lengthkey"],self.spec_prop.get("wspix", len(self.rawdata.T)))
                hduP.header.set('%s1'%self._build_properties["stepkey"],self.spec_prop.get("wstep", 1))
                hduP.header.set('%s1'%self._build_properties["startkey"],self.spec_prop.get("wstart",0))
            
        if naxis<3:
            hdul.append(ImageHDU([v for i,v in self.spaxel_mapping.items()], name='MAPPING'))
            hdul.append(ImageHDU([i for i,v in self.spaxel_mapping.items()], name='SPAX_ID'))
            hdul.append(ImageHDU(self.spaxel_vertices, name='SPAX_VERT'))
            # Spaxel Shape
            #e3d_group = PrimaryHDU(self.data, None)
            #hdul.append(ImageHDU([v for i,v in self.spaxel_mapping.items()], name='E3D_GRP'))
            
        hdulist = HDUList(hdul)
        hdulist.writeto(savefile,overwrite=force)

    @classmethod
    def read(cls, filename, format=None, **kwargs):
        """ """
        if format is None:
            format = filename.split(".")[-1]

        if format == "fits":
            return cls.read_fits(filename)
        
        if format in ["h5","hdf","hdf5"]:
            return cls.read_hdf(filename)
        
        raise NotImplementedError(f"Only fits and hdf5 format implemented, {format} given")

    @classmethod
    def read_fits(cls, filename, dataindex=0, headerindex=None, **kwargs):
        """ """
        this  = cls(None)
        this.load(filename, format="fits", dataindex=dataindex, headerindex=headerindex, **kwargs)
        return this

    @classmethod
    def read_hdf(cls, filename, **kwargs):
        """ """
        this  = cls(None)
        this.load(filename, format="hdf", **kwargs)
        return this

    def load(self, filename, format=None, **kwargs):
        """ """
        if format is None:
            format = filename.split(".")[-1]

        if format == "fits":
            self.load_fits(filename)
        
        elif format in ["h5","hdf","hdf5"]:
            self.load_hdf(filename)
        else:
            raise NotImplementedError(f"Only fits and hdf5 format implemented, {format} given")
                      
    def load_fits(self, filename, dataindex=0, headerindex=None):
        """  load the data cube. 
        Several format have been predifed. 
        
        lbda - If an hdu column of the fits file is name:
               "LBDA" or "LAMBDA" or "WAVE" or "WAVELENGTH" or "WAVELENGTHS",
               the column will the used as lbda
               
        // For non 3D cubes

        spaxel_mapping - looks for a column named "MAPPING" or "POS"
            This column contains the [x,y] coordinates of the spaxels
            In addition this looks for a column named "INDEXES" or "SPAX_ID"
            that contains the spaxels id. If not the spaxels will be order 0,1...N
        
        spaxel_vertices - looks for a column name "SPAX_VERT" that contains
            the vertices of the spaxels. 
            (not mandatory)
        
        
        """
        from astropy.io.fits import open as fitsopen
        self._side_properties["filename"] = filename
        self._side_properties["fits"]     = fitsopen(filename)
        
        if headerindex is None:
            headerindex = dataindex
            
        # Get the data
        data = self.fits[dataindex].data
        
        # Get the variance or the error
        variance = [f.data for f in self.fits if f.name.upper() in ["VAR",'VARIANCE', "VARIANCES"]]
        if len(variance)==0:
            variance = [f.data**2 for f in self.fits if f.name.upper() in ["ERR","ERROR", "ERRORS"]]
        if len(variance)==0:
            variance= None
        else:
            variance = variance[0]
            
        # Get the LBDA if any
        lbda_ = [f.data for f in self.fits if f.name.upper() in ["LBDA","LAMBDA", "WAVE", "WAVELENGTH","WAVELENGTHS"]]
        lbda = None if len(lbda_)==0 else lbda_[0]

        # = Spaxel Positions
        naxis = self.fits[headerindex].header.get("NAXIS",None)
        if naxis is not None and naxis <3:
            # euro3d format
            mapping = [f.data for f in self.fits if f.name.upper() in ["MAPPING", "POS"]]
            indexes = [f.data for f in self.fits if f.name.upper() in ["INDEXES","SPAX_ID"]]
            
            if len(mapping) ==0:
                raise TypeError("Could not find the SPAXEL MAPPING entry (MAPPING)")
            mapping = mapping[0]
            if len(indexes) ==0:
                warnings.warn("No indexes found for the spaxel mapping. arange used (0,1,...N)")
                indexes = np.arange(len(mapping))
            else:
                indexes = indexes[0]
                
            spaxel_mapping = {i:v.tolist() for i,v in zip(indexes, mapping)}
        else: # most likely 3D format
            spaxel_mapping= None
            
        # = Spaxel Shape
        spaxels_vertices = [f.data for f in self.fits if f.name.upper() in ["SPAX_VERT"]]
        if len(spaxels_vertices) >0:
            self.set_spaxel_vertices(spaxels_vertices[0])
            
        # --- Create the object
        self.create(data=data, header=self.fits[headerindex].header,
                    variance=variance, lbda=lbda, spaxel_mapping=spaxel_mapping)


    def load_hdf(self, filename):
        """ """
        import pandas
        from astropy.io import fits
        # Data
        data_ = pandas.read_hdf(filename, "data")
        indexes = np.asarray(data_.index)
        data = data_.values.T
        # Variance
        try:
            variance = pandas.read_hdf(filename, "variance").values.T
        except:
            variance = None
            
        # lbda
        lbda = pandas.read_hdf(filename, "lbda").values
        header = fits.Header( dict(pandas.read_hdf(filename, "header")) )
        spaxel_mapping_ = pandas.read_hdf(filename, "spaxel_mapping").values
        spaxel_mapping  = {k:v for k,v in zip(indexes, spaxel_mapping_)}
        spaxel_vertices = pandas.read_hdf(filename, "spaxel_vertices").values

        self.set_spaxel_vertices(spaxel_vertices)
        self.create(data=data, header=header,
                    variance=variance, lbda=lbda, spaxel_mapping=spaxel_mapping)
        
        
    # Old e3d format
    @staticmethod
    def load_from_e3d(e3dfile):
        """ Load a Cube stored as an old, historical e3d format.
        
        Returns
        -------
        pyifu cube.
        """
        from astropy.io import fits
        if not is_old_e3dformat(e3dfile):
            raise IOError("load_from_e3d only accepts old e3d format This is not one.")

        fitsfile = fits.open(e3dfile)
        d1, d2 = fitsfile[1:3]
        # Spaxel Shape:
        if d2.data["G_SHAPE"] == "S":
            spaxel_vertices = (np.asarray([[0,0],[0,1],[1,1],[1,0]]) - (0.5,0.5))
            spaxel_size = d2.data["G_SIZE1"]
            if d2.data["G_ANGLE"] !=0:
                angl = d2.data["G_ANGLE"]*np.pi/180
                rot_matrix = np.asarray([[np.cos(angl), np.sin(angl)],[-np.sin(angl),np.cos(angl)]])
                spaxel_vertices = np.dot(spaxel_vertices,rot_matrix)
        else:
            raise NotImplemented("Only square format implemented")

        # Data, Variance and Spaxel Mapping
        size     = d1.data["SPEC_LEN"][0]
        data, variance = [],[]
        spaxel_mapping = {}
        for i, d_, var_,x_,y_ in zip(np.sort(d1.data["SPEC_ID"]),
                                         d1.data["DATA_SPE"],d1.data["STAT_SPE"],
                                        d1.data["XPOS"]/spaxel_size, d1.data["YPOS"]/spaxel_size):
            if len(d_)!=size:
                continue
            data.append(d_)
            variance.append(var_)
            spaxel_mapping[i] = [x_,y_]

        data = np.asarray(data, dtype="float").T
        variance = np.asarray(variance, dtype="float").T
        # Wavelength
        lbda = d1.header['CDELTS']* np.arange(size) + d1.header['CRVALS']

        # header
        header = d1.header
        #
        # Creating the cube and returning it
        #
        cube = Cube(None)
        cube.create(data, header=header, variance=variance,
                    lbda=lbda, spaxel_mapping=spaxel_mapping)
        if spaxel_vertices is not None:
            cube.set_spaxel_vertices(spaxel_vertices)
        cube.spaxel_size = spaxel_size
        cube._side_properties["filename"] = e3dfile
        cube._side_properties["fits"]     = fitsfile

        return cube


    def contains(self, x, y, buffer=None):
        """ Test if the x and y coordinates (in spaxels units) are withing the contour polygon """
        from shapely import vectorized
        poly = self.contour_polygon if buffer is None else self.contour_polygon.buffer(buffer)
        return vectorized.contains(poly, x,y)

    def to_pandas(self):
        """ """
        import pandas
        data = pandas.DataFrame(self.data.T, index=self.indexes)
        if self.has_variance():
            var  = pandas.DataFrame(self.variance.T, index=self.indexes)
        else:
            var = None
        lbda  = pandas.Series(self.lbda)
        mapping = pandas.DataFrame( self.spaxel_mapping, index=["x","y"]).T
        verts = pandas.DataFrame(self.spaxel_vertices, columns=["x","y"])
        header = pandas.Series( dict(self.header))
        return {"data":data, "variance":var, "lbda":lbda, 
                "spaxel_mapping":mapping, "spaxel_vertices":verts, "header":header}
        
    
    def get_spaxels_within_polygon(self, polygon, inclusive=True):
        """ Returns the spaxel within the given polygon. 
        (this uses fraction_of_spaxel_within_polygon, 
        which given a weight between 0 and 1
        corresponding to the fraction of a given spaxel 
        contains within the polygon: 0 out, 1 all in)
        
        = This method uses the Shapely library =

        Parameters
        ----------
        polygon: [Shapely's Polygon]
            Are the spaxels within this polygon?
        
        inclusive: [bool] -optional-
            If a spaxel is partially within the polygon (weight<1), what to do:
            True: It is returned
            False: It is not returned
 
        Returns
        -------
        list of indexes
        """
        w = self.fraction_of_spaxel_within_polygon(polygon)
        if inclusive:
            return np.asarray(self.indexes[np.asarray(w, dtype="bool")]).tolist()
        return np.asarray(self.indexes[np.asarray(np.asarray(w, dtype="int"), dtype="bool")]).tolist()

    def get_spaxels_contours(self, indexes):
        """ """
        from shapely import geometry, ops
        mpoly = geometry.MultiPolygon([geometry.Polygon(self.spaxels_vertices[k]).buffer(0.1) for k in indexes])
        return ops.cascaded_union(mpoly)
 
    def fraction_of_spaxel_within_polygon(self, polygon):
        """ get the fraction of spaxel overlapping with the given polygon
        0 means all out and 1 means all in
        
        Parameters
        ----------
        polygon: [Shapely's Polygon]
            Are the spaxels within this polygon?
        
        Returns
        -------
        array of float between 0 and 1
        """
        
        from shapely.vectorized import contains
        from shapely.geometry   import Polygon
        
        polyvert  = self.spaxels_vertices # np.asarray([verts for verts in np.asarray([self.spaxel_vertices+np.asarray(self.index_to_xy(id_)) for id_ in self.indexes])])
        # xs and ys are the x and y coordinates of all the vertices
        xs, yx    = polyvert.T
        spaxel_edges_in = contains(polygon, xs,yx)
        # flagin
        flagin    = np.any(spaxel_edges_in, axis=0)
        flagallin = np.all(spaxel_edges_in, axis=0)
        flagpartial = flagin * ~flagallin
        # fraction of the 
        weights_in = np.zeros(self.nspaxels)
        weights_in[flagallin] = 1
        spaxels_area = Polygon(self.spaxel_vertices).area
        weights_in[flagpartial] = [polygon.intersection(Polygon(vert_partial)).area/spaxels_area for vert_partial in np.asarray(polyvert[flagpartial])]
        
        return weights_in

    def get_aperture(self, x, y, radius,
                    ell=None, theta=0, radius_min=None, **kwargs):
        """ 
        **kwargs goes to cube.get_slice

        theta: [float] -optional-
           Angle from the x-axis. [in radian]

        radius_min:[float] - optional-
            
        """
        
        import shapely
        # - Aperture per slice
        aperture = shapely.geometry.Point(x,y).buffer(radius)
        if radius_min is not None:
            aperture = aperture.difference(shapely.geometry.Point(x,y).buffer(radius_min))
            
        if ell is not None or ell == 0:
            aperture = shapely.affinity.scale( aperture, xfact=1, yfact=1-ell, origin="center")
            aperture = shapely.affinity.rotate(aperture, theta, origin='center', use_radians=True)
            
        wmap  = self.fraction_of_spaxel_within_polygon(aperture)
        return np.nansum(self.data*wmap), np.nansum(self.variance*wmap**2), np.nansum(wmap)

    def rotate(self, angle, rot_centroid=[0,0], inplace=False):
        """ returns a copy of this object but rotated.

        Parameters:
        -----------
        angle: float
            rotation angle in radian.
            N.B: positive means natural direction, i.e. counter clock wise.

        rot_centroid: [float,float] -optional-
            Coordinate (x,y) of the rotation axis. 

        inplace: [bool] -optional-
            A rotated copy of the current object will be returned
            except if inplace is set to True. If so, the current object will
            be affected by the rotation and nothing will be retured.

        Returns
        -------
        rotated Copy (Void if inplace=True, see doc.)
        """
        # What should it affect ?
        sp_ = self.copy() if not inplace else self
        # Rotate the centroids and the vertices
        spaxels_xy = np.asarray( self.index_to_xy(self.indexes) )
        rotmat     = np.asarray([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        spaxels_xy = np.dot(rotmat, (spaxels_xy-rot_centroid).T).T + rot_centroid
        vertices   = np.dot(rotmat, self.spaxel_vertices.T).T
        # Set it back in
        sp_.set_spaxel_mapping({i:c_ for i,c_ in zip(self.indexes, spaxels_xy)})
        sp_.set_spaxel_vertices(vertices)
        # Return if needed
        if not inplace:
            return sp_

    def get_index_vertices(self, index):
        """ """
        return np.asarray([self.spaxel_vertices + np.asarray( self.index_to_xy(index_, fill_with=np.NaN) )
                               for index_ in np.atleast_1d(index)])
    
    def index_to_xy(self, index, fill_with=None):
        """ Each spaxel has a unique index (1d-array) entry.
        This tools enables to know what is the 2D (x,y) position 
        of this index

        Returns
        -------
        [int,int] (x,y)
        """
        if is_arraylike(index):
            return [self.index_to_xy(index_) for index_ in index]
        
        if index in self.spaxel_mapping:
            return self.spaxel_mapping[index]
        return fill_with, fill_with

    def xy_to_index(self, xy):
        """ Each spaxel has a unique x,y, location that 
        can be converted into a unique index (1d-array) entry.
        This tools enables to know what is the index corresponding to the given
        2D (x,y) position.
        
        Parameters
        ----------
        xy: [2d array]
            x and y position(s) in the following structure:
            [x,y] or [[x0,y0],[x1,y1]]

        Returns
        -------
        list of indexes 
        """
        # - Following set_spaxels_mapping, v are list
        # make sure xy too:
        xy = np.asarray(xy).tolist()
        if np.shape(xy) == (2,):
            return [i for i,v in self.spaxel_mapping.items() if v == xy]
        return [i for i,v in self.spaxel_mapping.items() if np.any(v in xy)]

    def is_spaxels_nan(self):
        """ """
        return np.asarray(np.sum(np.isnan(self.spaxels_vertices), axis=(1,2)), dtype="bool")
    
    def get_spaxel_polygon(self, remove_nan=False, format="polygons"):
        """ return list of Shapely Polygon corresponding to the spaxel position on sky 
        
        format: [string] -optional-
            output format:
            - polygons: list of polygons
            - multipylygon: MultiPolygon
            - list: list of vertices
            - array: array of vertices
            - dict: dictionary {index:vertex}
            - dataframe: pandas.DataFrame(vectex, name: index)

        """
        
        x,y = self.xy
        all_poly = self.spaxels_vertices
        if remove_nan:
            flagout = self.is_spaxels_nan()
            all_poly = all_poly[~flagout]
            
        if "polygon" in format:
            
            #from shapely import geometry
            
            if format.lower() == "polygons":
                from shapely.geometry import Polygon
                return [Polygon(p) for p in all_poly]
            elif format.lower() in ["multipolygon", "multipolygons"]:
                from shapely.geometry import Polygon, MultiPolygon
                return MultiPolygon([Polygon(p) for p in all_poly])
            raise ValueError("Cannot parse the given format (%s)"%format)
        
        if format in ["list"]:
            return all_poly
        if format in ["array"]:
            return np.asarray(all_poly)
        if format in ["dict"]:
            indexes = self.indexes if not remove_nan else self.indexes[~flagout]
            return {indexes[i]:p for i,p in enumerate(all_poly)}
                        
        raise ValueError("Cannot parse the given format (%s)"%format)

    def display_contours(self, ax=None, buffer=0.001, facecolor="None", edgecolor="0.7",
                             zorder=None, autoscale=True, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        from matplotlib import patches
        from shapely import geometry
        from shapely.ops import cascaded_union

        if ax is None:
            fig = mpl.figure(figsize=[6,5])
            ax  = fig.add_subplot(111)
        else:
            fig = ax.figure

        
        spaxels = [geometry.Polygon(v_).buffer(buffer) for v_ in self.get_index_vertices(self.indexes) if not np.any(np.isnan(v_))]
    
        MLA = cascaded_union(spaxels)
        vertices = np.asarray(MLA.exterior.xy).T
        # Add it to SDSS image
        p = patches.Polygon( vertices, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, **kwargs)
        ax.add_patch(p)
        
        if autoscale:
            ax.autoscale(True, tight=True)
            
        return MLA
    
    # =================== #
    #   Properties        #
    # =================== #
    def is_3d_cube(self):
        """ """
        return len(self.data.shape) == 3

    @property
    def xy(self):
        """ position of the center of each spaxel (in spaxel units)"""
        return np.asarray(self.index_to_xy(self.indexes)).T
    
    @property
    def spaxel_mapping(self):
        """ dictionary containing the connection between spaxel index (int) and spaxel position (x,y)"""
        if self._properties["spaxel_mapping"] is None:
            self._properties["spaxel_mapping"] = {}
        return self._properties["spaxel_mapping"]
    
    def _header_to_spec_prop_(self):
        """ """
        self.spec_prop["logwave"]= self.header.get('logwave', None)
        if self.header.get("NAXIS") ==2:
            self.spec_prop["wspix"]  = self.header.get('%s1'%self._build_properties["lengthkey"])
            self.spec_prop["lspix"]  = self.header.get('%s2'%self._build_properties["lengthkey"])
            self.spec_prop["wstep"]  = self.header.get('%s1'%self._build_properties["stepkey"])
            self.spec_prop["lstep"]  = self.header.get('%s2'%self._build_properties["stepkey"])
            self.spec_prop["wstart"] = self.header.get('%s1'%self._build_properties["startkey"])
            self.spec_prop["lstart"] = self.header.get('%s2'%self._build_properties["startkey"])
        else:            
            self.spec_prop["wspix"]  = self.header.get('%s1'%self._build_properties["lengthkey"])
            self.spec_prop["nspix"]  = self.header.get('%s2'%self._build_properties["lengthkey"])
            self.spec_prop["lspix"]  = self.header.get('%s3'%self._build_properties["lengthkey"])

            self.spec_prop["wstep"]  = self.header.get('%s1'%self._build_properties["stepkey"])
            self.spec_prop["nstep"]  = self.header.get('%s2'%self._build_properties["stepkey"])
            self.spec_prop["lstep"]  = self.header.get('%s3'%self._build_properties["stepkey"])

            self.spec_prop["wstart"] = self.header.get('%s1'%self._build_properties["startkey"])
            self.spec_prop["nstart"] = self.header.get('%s2'%self._build_properties["startkey"])
            self.spec_prop["lstart"] = self.header.get('%s3'%self._build_properties["startkey"])

    @property
    def nspaxels(self):
        """ Number of spaxel recorded in the spaxel mapping """
        return len(self.spaxel_mapping)
    
    @property
    def indexes(self):
        """ Name/ID of the spaxels. (keys from spaxel_mapping)"""
        return np.sort( list(self.spaxel_mapping.keys())) if self.spaxel_mapping is not None else []
    
    @property
    def spaxel_mapping(self):
        """ dictionary containing the connection between spaxel index (int) and spaxel position (x,y)"""
        return self._properties["spaxel_mapping"]

    @property
    def spaxel_vertices(self):
        """ The reference vertices for the spaxels. """
        if self._properties["spaxel_vertices"] is None:
            self.set_spaxel_vertices([[0.5, 0.5],[-0.5,0.5],[-0.5, -0.5],[0.5,-0.5]])
        return self._properties["spaxel_vertices"]

    @property
    def spaxels_vertices(self):
        """ """
        return self.xy.T[:,None] + self.spaxel_vertices
    @property
    def contour_polygon(self):
        """ """
        from shapely import geometry
        return geometry.MultiPolygon( self.get_spaxel_polygon(remove_nan=True) ).convex_hull
    
    @property
    def convexhull_vertices(self):
        """ """
        return np.asarray(self.contour_polygon.exterior.xy)
    
class Slice( SpaxelHandler ):
    """ """
    # ================= #
    #  I/O              #
    # ================= #
    
    @classmethod
    def from_cubefile(cls, filename, lbdarange=[6000,8000]):
        """ """
        from astropy.io import fits
        with fits.open(filename) as f:
            # Loading header and lbda
            header = f["PRIMARY"].header
            lbda = np.arange(header["NAXIS2"])*header["CDELT1"] + header["CRVAL1"]
            flagin = (lbda>lbdarange[0]) & (lbda<lbdarange[1])
            # Data
            data = np.nanmean(f["PRIMARY"].data.copy()[flagin], axis=0)
            variance = np.nanmean(f["VARIANCE"].data.copy()[flagin], axis=0)
            # spaxel mapping
            spaxel_mapping = {i:k for i,k in zip(f["SPAX_ID"].data,f["MAPPING"].data)}
            spaxel_vert    = f["SPAX_VERT"].data
            spaxel_vert = f["SPAX_VERT"].data.copy()
            # loading the cls
            slice_ = cls(None)
            slice_.create(data, variance=variance, lbda=np.mean(lbda[flagin]),
                              spaxel_mapping=spaxel_mapping.copy())
            slice_.set_spaxel_vertices(spaxel_vert.copy())
            slice_.set_header(header.copy())
            # cleaning the tmp variables
            del header, lbda, flagin
            del data, variance
            del spaxel_mapping, spaxel_vert
    
        return slice_

    # ================= #
    #  Main Method      #
    # ================= #
    # // get some spaxels
    def get_brightest_spaxels(self, nspaxels=1):
        """ """
        return self.indexes[np.argsort(self.data)][::-1][:nspaxels]
    
    def get_faintest_spaxels(self, nspaxels=1):
        """ """
        return self.indexes[np.argsort(self.data)][:nspaxels]
    
    # //
    def get_subslice(self, indexes):
        """ """
        slice_ = self.__class__(None)
        spaxelsin = np.in1d(self.indexes, indexes)
        slice_.create(self.data[spaxelsin],
                      variance=self.variance[spaxelsin] if self.has_variance() else None,
                     lbda=self.lbda,
                     spaxel_mapping={k:v for k,v in self.spaxel_mapping.items() if k in indexes})
        slice_.set_spaxel_vertices(self.spaxel_vertices)
        return slice_
        
        
    
    def show(self, toshow="data", ax = None, savefile=None, show=True,
                 vmin=None, vmax=None, show_colorbar=True, cmap=None,
                 clabel="",cfontsize="large", empty_if_nan=True,
                 autoscale=True,
                 alpha=0.8, ec="0.5", lw=0, **kwargs):
        """ display on the IFU hexagonal grid the given values
        
        Parameters
        ----------
        """
        from matplotlib import patches
        from .tools import figout, mpl

        # -- Let's go
        if ax is None:
            fig = mpl.figure(figsize=[6,5])
            ax  = fig.add_subplot(111)
        else:
            fig = ax.figure

        value = np.asarray( eval("self.%s"%toshow) ) if toshow is not None else None
        if value is None or np.all(np.isnan(value)):
            show_colorbar = False
        else:
            cmap = mpl.get_cmap(cmap)
            # - which colors
            if vmin is None:
                vmin = np.percentile(value[value==value],0)
            elif type(vmin) == str:
                vmin = np.percentile(value[value==value],float(vmin))
            if vmax is None:
                vmax = np.percentile(value[value==value],100)
            elif type(vmax) == str:
                vmax = np.percentile(value[value==value],float(vmax))
                
            colors = cmap((value-vmin)/(vmax-vmin))
            
                
        x,y = self.xy
    
        # - The Patchs
        for i in range(self.nspaxels):
            c_ = "None" if (value is None or (np.isnan(value[i]) and empty_if_nan)) else colors[i]
            linewidth = 1 if (type(c_) == str and c_ == "None" and lw==0) else lw
            ax.add_patch(patches.Polygon(self.spaxel_vertices+np.asarray([x[i],y[i]]),
                                facecolor=c_, alpha=alpha, linewidth=linewidth, edgecolor=ec,
                                  **kwargs))

        if show or autoscale:
            ax.autoscale(True, tight=True)
            
        if show_colorbar:
            from .tools import colorbar, insert_ax
            axcbar = ax.insert_ax("right", shrunk=0.88)
            axcbar.colorbar(cmap, vmin=vmin, vmax=vmax, label=clabel,
                    fontsize=cfontsize)
    
        fig.figout(savefile=savefile, show=show)
        return fig

        
class Cube( SpaxelHandler ):
    """
    This Class is the basic class upon which the other Cube will be based.
    In there, there is just the basic method.
    """
    SIDE_PROPERTIES = ["adr"]
    
    # ================ #
    #  Main Method     #
    # ================ #
    
    # -------- #
    # Oper.    #
    # -------- #
    def _check_other_(self, other, kind="standard operation (add, div, sub etc.)"):
        """ Raise a Type Error if other cannot to standard operatio """
        if Cube not in other.__class__.__mro__:
            raise TypeError("Cannot do %s between a cube with a non-Cube object"%kind)
        if np.shape(self.data) != np.shape(other.data):
            raise TypeError("Cannot do `%s` between cubes with different data shape"%kind)
        
    def __add__(self, other):
        """ """
        self._check_other_(other, kind="addition")
        
        data = self.data + other.data
        var1 = 0 if not self.has_variance() else self.variance
        var2 = 0 if not other.has_variance() else other.variance
        var  = np.sqrt(var1 + var2)
        if len(np.atleast_1d(var)) == 1: var = None
        return get_cube(data, lbda=self.lbda, variance=var, header=self.header,
                        spaxel_mapping=spaxel_mapping,
                        spaxel_vertices=self.spaxel_vertices)

    def __sub__(self, other):
        """ """
        self._check_other_(other, kind="subtraction")
        
        data = self.data - other.data
        var1 = 0 if not self.has_variance() else self.variance
        var2 = 0 if not other.has_variance() else other.variance
        var  = np.sqrt(var1 + var2)
        if len(np.atleast_1d(var)) == 1: var = None
        return get_cube(data, lbda=self.lbda, variance=var, header=self.header,
                        spaxel_mapping=self.spaxel_mapping,
                        spaxel_vertices=self.spaxel_vertices)

    def __div__(self, other, kind="division"):
        """ """
        data = self.data / other.data
        var1 = None if not self.has_variance() else self.variance
        var2 = None if not other.has_variance() else other.variance
        if var1 is None and var2 is None:
            var = None
        elif var1 is None:
            var = var2
        elif var2 is None:
            var = var1
        else:
            var  = data*np.sqrt(self.data**2/var1 + other.data**2/var2)
            
        return get_cube(data, lbda=self.lbda, variance=var, header=self.header,
                        spaxel_mapping=self.spaxel_mapping,
                        spaxel_vertices=self.spaxel_vertices)
        
    # -------- #
    #   I/O    #
    # -------- #

    # --------- #
    #  SLICES   #
    # --------- #
    def get_slice(self, lbda_trans=None,
                      lbda_min=None, lbda_max=None, index=None,
                      usemean=True, data="data",
                      slice_object=False):
        """ Returns a array containing the [weighted] average for the 
        spaxels within the given range.
        
        
        Parameters
        ----------
        lbda_trans: [None or [float, float] -optional-
            Provide here the wavelength and transmission of the filter you want to apply
            to the data. 
            i.e: lbda_trans = [lbda_filter, trans_filter]

        lbda_min, lbda_max: [None or float] -required if not index-
            lower and higher boundaries of the wavelength (in Angstrom) defining the slice.
            None means no limit.
            ```flagin = (self.lbda>=lbda_min) * (self.lbda<=lbda_max)```
            *Ignored if index provided*
            
        index: [int] -optional-
            Index of the cube slice you want. 

        usemean: [bool] -optional-
            If the cube has a variance, the slice will be the weighted mean (weighted by inverse of variance) 
            of the slice value within the wavelength range except if 'usemean' is set to True. 
            In that case, the mean (and not weigthed mean) will be used.
            
        data: [string] -optional-
            Variable that will be used.
            If `data`  do not contains 'data' in its name usemean will be forced to False.
            Do not change this is you have a doubt.
            
        slice_object: [bool] -optional-
            Shall this returns a Slice object or jsut the corresponding data?

        Returns
        -------
        array of float
        """
        if lbda_trans is not None:
            try:
                lbda_f, trans_f = lbda_trans
            except:
                raise ValueError("lbda_trans must be a 2D array if not None; lbda_f, trans_f = lbda_trans")
            slice_data = synthesize_photometry3d(self.lbda, self.data, lbda_f, trans_f)
            slice_var  = synthesize_photometry3d(self.lbda, self.variance, lbda_f, trans_f)
            lbda = np.average(lbda_f, weights=trans_f)
    
        elif index is not None:
            slice_data = eval("self.%s[%d]"%(data,index))
            slice_var  = None if data not in ['data','rawdata'] or not self.has_variance()\
              else eval("self.variance[%d]"%(index))
            lbda = self.lbda[index]
        else:
            if lbda_min is None: lbda_min = self.lbda[0]
            if lbda_max is None: lbda_max = self.lbda[-1]
                
            if lbda_min>lbda_max: lbda_min,lbda_max = np.sort([lbda_min,lbda_max])
                
            flagin = (self.lbda>=lbda_min) * (self.lbda<=lbda_max)
            flagin = np.asarray(flagin, dtype="bool")
            lbda=np.mean(self.lbda[flagin])
            if 'data' not in data or not self.has_variance():
                usemean   = True
                slice_var = None
            else:
                slice_var = np.nanmean( eval("self.%s"%"variance")[flagin], axis=0)
                
            if not self.has_variance() or usemean:
                slice_data = np.asarray([np.nanmean(self.get_index_data(i, data=data)[flagin])
                                             for i in range(self.nspaxels)])
            else:
                slice_data = np.asarray([np.average(self.get_index_data(i, data=data)[flagin],
                                        weights=1./self.get_index_data(i, data="variance")[flagin])
                                    for i in range(self.nspaxels)])
            
        if not slice_object:
            return slice_data

        return get_slice(slice_data, self.index_to_xy(self.indexes), spaxel_vertices=self.spaxel_vertices, 
                                 variance=slice_var, indexes=self.indexes, lbda=lbda)

    def to_metaslices(self, lbda_ranges, nslices, as_slice=False):
        """ Split Cube into N metaslices.

        Parameters
        ----------
        lbda_ranges: list or array of 2 floats
            lbda_min and lbda_max of the desired wavelength range

        nslices: int
            Number of desired metaslices in lbda_ranges

        as_slices: bool
            If True (Default), return *metaslices* Slices() object.
            If False return 3 arrays (data, var, lbda)

        Returns
        -------
        Slice() or array
        """
        STEP_LBDA_RANGE = np.linspace(lbda_ranges[0],lbda_ranges[1], nslices+1)
        lbda_stepbin = np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T

        if not as_slice:
            binned_data = np.zeros( (nslices, np.shape(self.data)[-1] ))
            if self.has_variance():
                binned_var = np.zeros( (nslices, np.shape(self.variance)[-1] ))
            else:
                binned_var = None
            binned_lbda = np.zeros( nslices )
        else:
            slices=[]

            
        for (i,j) in enumerate( lbda_stepbin ):
            slice_obj = self.get_slice(lbda_min=j[0], lbda_max=j[1], slice_object=True)
            if not as_slice:
                binned_data[i,:] = slice_obj.data
                if self.has_variance():
                    binned_var[i,:] = slice_obj.variance
                    
                binned_lbda[i] = slice_obj.lbda
            else:
                slices.append(slice_obj)

        if not as_slice:
            return ( binned_data, binned_var, binned_lbda )

        elif as_slice:
            return slices

    def to_metacube(self, lbda_ranges, nslices=5):
        """ Fuse slices of self to get meta-cube.

        Parameters
        ----------
        lbda_ranges: list or array of 2 floats
            lbda_min and lbda_max of the desired wavelength range

        nslices: int
            Number of desired metaslices in lbda_ranges

        Returns
        -------
        Cube()
        """
        data, var, lbda = self.to_metaslices( lbda_ranges=lbda_ranges, nslices=nslices, as_slice=False)   
        metacube = self.get_new( newdata=data, newlbda=lbda, newvariance=var if var is not None else "None")

        return metacube    
    # --------- #
    #  SPAXEL   #
    # --------- #
    # - GET SPECTRUM
    def get_spectrum(self, index, data="data", usemean=False, spaxelindexes=False):
        """ Return a Spectrum object based on the given index
        If index is a list of indexes, the mean spectrum will be returned.
        (see xy_to_index to convert 2D coordinates into index)

        Parameters
        ----------
        index: [int, list of]
            index (or list of index) for which you want a spectrum.
            If a list is given, the returned spectrum will be the average spectrum.

        data: [string] -optional-
            Which attribute of the cube will be considered as the 'data' of the cube?

        usemean: [bool] -optional-
            If several indexes are given, the mean spectrum will be returned.
            If the variance is available and does not contains NaN, the weighted (1/variance)
            average will be used to combine spectra except if `usemean` is True. 
            In that case, the simple mean will be used.
        
        spaxelindexes: [bool] -optional-
            If True, this means you gave index as stored in self.indexes, and not as entries of self.indexes. 
            i.e.: if self.indexes = [12,18,22,23;26,40...].and you want to use spaxels associated indexes [12, 18, 23]:
            do:
                - index=[0,1,3],      spaxelindexes=False
             or - index=[12, 18, 23], spaxelindexes=True
            
        Returns
        -------
        Spectrum
        """
        spec = Spectrum(None)

        data_ = self.get_index_data(index, data, spaxelindexes=spaxelindexes)
        variance = self.get_index_data(index, data="variance", spaxelindexes=spaxelindexes) if self.has_variance() and "data" in data else None
        
        if is_arraylike(index):
            # multiple indexes
            if not usemean and variance is not None and not np.isnan(variance).any():
                data_ = np.average(data_, weights = 1./np.asarray(variance), axis=0)
            else:
                data_ = np.nanmean(data_, axis=0)
                
            if variance is not None:
                variance = np.nanmean(variance, axis=0) / len(index)

        spec.create(data=data_, variance=variance, header=None, lbda=self.lbda)
        return spec
    
    # - PARTIAL CUBE
    def get_partial_cube(self, indexes, slice_id, verbose=False):
        """ Get a subsample of the current cube.
        The return cube will only contain the requested slices (spaxel positions) 
        at the requested wavelengths

        Parameters
        ----------
        indexes: [list or boolean-array]
            Indexes (as in spaxel_mapping) of the spaxels you want to keep.

        slice_id: [list or boolean-array]
            Id (from 0->NLBDA) of the slices you want to keep
            = Info this will be selected as self.lbda[slice_id] =

        Returns
        -------
        Cube
        """
        spaxelsin = np.in1d(self.indexes, indexes)
        if verbose: print(np.argwhere(spaxelsin))
            
        rawdata = self.rawdata[slice_id].T[spaxelsin].T
        if self.has_variance():
            var = self.variance[slice_id].T[spaxelsin].T
        else:
            var = None

        spaxel_mapping = {id_:self.spaxel_mapping[id_] for id_ in indexes}
        copy_cube = self.copy()
        
        copy_cube.create(rawdata, variance=var, header=self.header,
                        lbda=self.lbda[slice_id],
                        spaxel_mapping=spaxel_mapping)
        copy_cube.set_spaxel_vertices(self.spaxel_vertices)
        
        # - If you removed a flux, rawdata != data
        if self._derived_properties["data"] is not None:
            copy_cube._derived_properties["data"] = self._derived_properties["data"][slice_id].T[spaxelsin].T
        
        return copy_cube


    def get_sourcecube(self, sourcedf, sourcescale=5, slice_id=None):
        """ extract a partial cube containing the spaxels within the source provided in the sourcedf

        sourcedf: [pandas.DataFrame]
            this dataframe must contain the sep/sextractor ellipse information:
            x,y for the centroid
            a,b for the second moment (major and minor axis)
            theta for the angle (in degree)
            = must be in units of spaxels =
            
        sourcescale: [float] -optional-
            this multiply a and b. 1 means second moment (1 sigma)
            
        slide_id: [None or list] -optinal-
            select the wavelength for the new returned cube.

        Returns
        -------
        Cube
        """
        from matplotlib.patches import Ellipse
        from shapely import geometry 
        e = [Ellipse((d.x,d.y), d.a*sourcescale*2, d.b*sourcescale*2, d.theta*180/np.pi)
             for d in sourcedf.itertuples()]
        polys = [geometry.Polygon(e_.get_verts()) for e_ in e]
        spaxels = np.unique(np.concatenate([self.get_spaxels_within_polygon(poly_) for poly_ in polys]))

        if slice_id is None:
            slice_id  = np.arange( len(self.lbda) )

        return self.get_partial_cube(spaxels,  slice_id)
    
    # - SORTING TOOLS
    def get_sorted_spectra(self, lbda_range=None, data="data",
                               descending=False,
                               avoid_area=None, avoid_indexes=None, 
                               **kwargs):
        """ Returns the brightness-sorted spaxel indexes.

        Parameters
        ----------
        lbda_range: [2D array / None] -optional-
            This enable to define a weavelength zone where the brightness
            will be defined. If None, full wavelength. if None in any upper 
            or lower bound, (i.e. [None, x]) no upper or lower restriction
            respectively.

        data: [string] -optional-
            Attribute that will be looked at to build the slice used to sort 
            the spaxels.

        descending: [bool] -optional-
            True:  Brightest -> Faintest
            False: Faintest  -> Brightest
                       
        avoid_area: [3-floats/None] -optional-
            You can avoid an space of the Cube by defining here a position 
            (x,y) and a radius. avoid_area must be (x,y,r). 
            All spaxel inthere will be avoided in the spaxel sorting 
            If None, nothing will be avoided this way. 

        avoid_indexes: [array/None] 
            You can set a list of spaxel coords [idx_1, idx_2,...] these 
            spaxels will be avoided in the spaxel sorting (considered as Nan).
            If None, nothing will be avoided this way.
            NB: see the xy_to_index() method to convert x,y, coords into indexes
                        
        **kwargs goes to get_slice(): e.g. usemean

        Returns
        -------
        list of indexes (if nan, they are at the end)
        """
        from scipy.spatial import distance
        
        # ------------------- #
        #    Input checked    #
        # ------------------- #
        if lbda_range is None:
            lbda_range = [None,None]  
        # - user gave a wrong input
        elif len(lbda_range) != 2:
            raise ValueError("`lbda_range` must be None or a 2D array")
        
        # ------------------- #
        #  Data To be Sorted  #
        # ------------------- #
        mean_data   = self.get_slice(lbda_min=lbda_range[0],lbda_max=lbda_range[1], data=data, **kwargs)
        if descending:
            mean_data *=-1
            
        # ------------------- #
        #   Avoidance Area    #
        # ------------------- #
        avoid_indexes = [] if avoid_indexes is None else np.asarray(avoid_indexes).tolist()
            
        # -- Avoidance Area
        if avoid_area is not None and len(avoid_area) == 3:
            x,y,rad_ = avoid_area[0],avoid_area[1],avoid_area[2] # to save cpu
            dist_ = distance.cdist([[x,y]],np.asarray(self.index_to_xy(self.indexes)))[0]
            avoid_indexes = avoid_indexes+np.asarray(np.where(dist_<0)).flatten().tolist()
        # -- NaNs to avoid too
        #avoid_nans = [i for i in self.indexes if np.any(np.isnan(self.data[i]))]
        # ------------------- #
        #   Actual Sorting    #
        # ------------------- #
        return [i for i in np.argsort(mean_data) if i not in avoid_indexes]
        
    def get_brightest_spaxels(self, nspaxels, lbda_range=None, data="data",
                               avoid_area=None, avoid_indexes=None, 
                               **kwargs):
        """ Returns the `nspaxel`-brightness spaxel indexes.

        Parameters
        ----------
        nspaxels: [int]
            Number of spaxel indexes to be returned

        
        lbda_range: [2D array / None] -optional-
            This enable to define a weavelength zone where the brightness
            will be defined. If None, full wavelength. if None in any upper 
            or lower bound, (i.e. [None, x]) no upper or lower restriction
            respectively.

        data: [string] -optional-
            Attribute that will be looked at to build the slice used to sort 
            the spaxels.
                       
        avoid_area: [3-floats/None] -optional-
            You can avoid an space of the Cube by defining here a position 
            (x,y) and a radius. avoid_area must be (x,y,r). 
            All spaxel inthere will be avoided in the spaxel sorting 
            If None, nothing will be avoided this way. 

        avoid_indexes: [array/None] 
            You can set a list of spaxel coords [idx_1, idx_2,...] these 
            spaxels will be avoided in the spaxel sorting (considered as Nan).
            If None, nothing will be avoided this way.
            NB: see the xy_to_index() method to convert x,y, coords into indexes
                        
        **kwargs goes to get_slice(): e.g. usemean

        Returns
        -------
        list of indexes (if nan, they are at the end)
        """
        return self.get_sorted_spectra(lbda_range=lbda_range,
                                        avoid_area=avoid_area,
                                        avoid_indexes=avoid_indexes,
                                        descending=True,**kwargs)[:nspaxels]
            
    
    def get_faintest_spaxels(self, nspaxels, lbda_range=None, data="data",
                               avoid_area=None, avoid_indexes=None, 
                               **kwargs):
        """ Returns the `nspaxel`-brightness spaxel indexes.

        Parameters
        ----------
        nspaxels: [int]
            Number of spaxel indexes to be returned

        
        lbda_range: [2D array / None] -optional-
            This enable to define a weavelength zone where the brightness
            will be defined. If None, full wavelength. if None in any upper 
            or lower bound, (i.e. [None, x]) no upper or lower restriction
            respectively.

        data: [string] -optional-
            Attribute that will be looked at to build the slice used to sort 
            the spaxels.
                       
        avoid_area: [3-floats/None] -optional-
            You can avoid an space of the Cube by defining here a position 
            (x,y) and a radius. avoid_area must be (x,y,r). 
            All spaxel inthere will be avoided in the spaxel sorting 
            If None, nothing will be avoided this way. 

        avoid_indexes: [array/None] 
            You can set a list of spaxel coords [idx_1, idx_2,...] these 
            spaxels will be avoided in the spaxel sorting (considered as Nan).
            If None, nothing will be avoided this way.
            NB: see the xy_to_index() method to convert x,y, coords into indexes
                        
        **kwargs goes to get_slice(): e.g. usemean

        Returns
        -------
        list of indexes (if nan, they are at the end)
        """
        return self.get_sorted_spectra(lbda_range=lbda_range,
                                        avoid_area=avoid_area,
                                        avoid_indexes=avoid_indexes,
                                        descending=False,**kwargs)[:nspaxels]

    # -------------- #
    # Manage 3D e3D  #
    # -------------- #
    def get_index_data(self, index, data="data", spaxelindexes=False):
        """ Return the `data` corresponding the the ith index 

        Parameters
        ----------
        index: [int, list of]
            index (or list of index) for which you want a spectrum.
            If a list is given, the returned spectrum will be the average spectrum.

        data: [string] -optional-
            Which attribute of the cube will be considered as the 'data' of the cube?

        spaxelindexes: [bool] -optional-
            If True, this means you gave index as stored in self.indexes, and not as entries of self.indexes. 
            i.e.: if self.indexes = [12,18,22,23;26,40...].and you want to use spaxels associated indexes [12, 18, 23]:
            do:
                - index=[0,1,3],      spaxelindexes=False
             or - index=[12, 18, 23], spaxelindexes=True

        """
        indexes = index if not spaxelindexes else np.arange(self.nspaxels)[np.isin(self.indexes,index)]
        if self.is_3d_cube():
            if is_arraylike(indexes):
                return [self.get_index_data(index_,data=data) for index_ in indexes]
            x,y = self.index_to_xy(indexes)
            return eval("self.%s.T[x,y]"%data)

        return eval("self.%s.T[indexes]"%data)
        
    
    # -------------- #
    #  Manipulation  #
    # -------------- #
    def remove_flux(self, flux, remove_from="rawdata"):
        """
        This enalble to remove the given flux to all the
        spaxels of the cube.
        
        This is set to `data`

        Parameters
        ----------
        flux: [array]              
            The input flux that will be removed.

        Returns
        -------
        Void, affects the object (data, variance)
        """
        if len(flux) != len(self.lbda):
            raise ValueError("The given `spec` must have the size as the wavelength array")

        self._derived_properties["data"] = np.asarray(eval("self.%s.T"%(remove_from)) - flux).T
        
    def scale_by(self, coef, onraw=True):
        """ divide the data by the given scaling factor 
        If this object has a variance attached, the variance will be divided by the square of `coef`.
        Parameters
        ----------
        coef: [float or array of]
            scaling factor for the data 

        onraw: [bool] -optional-
            scaley from rawdata or from data (that could already be scaled)

        Returns
        -------
        Void, affect the object (data, variance)
        """
        data     = self.rawdata if onraw else self.data
        variance = self.rawvariance if onraw else self.variance
        
        
        if not is_arraylike(coef) or len(coef)==1 or len(coef) == self.data.shape[1] or np.shape(coef)==self.data.shape:
            self._derived_properties["data"]  = data / coef
            if self.has_variance():
                self._derived_properties["variance"]  = variance / coef**2
                
        elif len(coef) == self.data.shape[0]:
            self._derived_properties["data"]  = np.asarray(data.T / coef).T
            if self.has_variance():
                self._derived_properties["variance"]  = np.asarray(variance.T / coef**2).T
        else:
            raise ValueError("scale_by is not able to parse the shape of coef.", np.shape(coef), self.data.shape)
            
        
    # ================================ #
    # ==     External Tools         == #
    # ================================ #
    def show(self, toshow="data",
                 interactive=False, launch=True,
                 savefile=None, ax=None, show=True,
                 show_meanspectrum=True, cmap=None,
                 vmin=None, vmax=None, notebook=None,
                 **kwargs):
        """ Display the cube.
        
        Parameters
        ----------
        toshow: [string] -optional-
            Variable you want to display. anything accessible as self.`toshow` that 
            has the same size as the wavelength. 
            If toshow is data or rawdata (or anything containing 'data'), 
            the variance will automatically be added if it exists.
            Do not change this is you have a doubt.
            
        interactive: [bool] -optional- 
           Enable to interact with the plot to navigate through the cube.
           (this might depend on your matplotlib setup.)

        launch: [bool] -optional-
            Shall the interactive mode (if selected) be launched?
            If not, use iplot.launch(vmin=vmin, vmax=vmax, notebook=notebook)
            where iplot is what has been returned by show()

        cmap: [matplotlib colormap] -optional-
            Colormap used for the wavelength integrated cube (imshow).

        vmin, vmax: [float /string / None] -optional-
            Lower and upper value for the colormap. 
            3 Formats are available:
            - float: Value in data unit
            - string: percentile. Give a float (between 0 and 100) in string format.
                      This will be converted in float and passed to numpy.percentile
            - None: The default will be used (percentile 0.5 and 99.5 percent respectively).
            (NB: vmin and vmax are independent, i.e. one can be None and the other '98' for instance)
        show_meanspectrum: [bool] -optional-
            If True both a wavelength integrated cube (imshow) and the average spectrum 
            will be displayed. If not, only the wavelength integrated cube (imshow) will.

        ax: [matplotlib.Axes] -optional-
            Provide the axes where the spectrum and/or the wavelength integrated 
            cube  will be drawn. 
            See show_meanspectrum:
               - If True, 2 axes are requested so axspec, aximshow=ax
               - If False, 1 axes is needed, aximshow=ax 
            If None this will create a new axes inside a new figure
            
        savefile: [string/None] -optional-
            Would you like to save the data? If so give the name of this
            file where the plot will be saved.
            You can provide an extention (.pdf or .png) if you don't both the
            .pdf and .png will be created.

        show: [bool] -optional-
            If you do not save the data (see savefile), shall the plot be shown?

        notebook: [bool or None] -optional-
            Is this running from a notebook? 
            If True, the plot will be made using fig.show() if not with mpl.show()
            If None, this will try to guess.

        **kwargs goes to matplotlib's imshow 

        Returns
        -------
        Void
        """
        if interactive:
            from .mplinteractive import InteractiveCube
            iplot = InteractiveCube(self, fig=None, axes=ax, toshow=toshow)
            if launch:
                iplot.launch(vmin=vmin, vmax=vmax, notebook=notebook)
            return iplot

        # - Not interactive
        from .tools import figout, specplot, mpl
        # - Axis definition
        if ax is None:
            fig = mpl.figure(figsize= [6,5] if not show_meanspectrum else [10,3.5] )
            if show_meanspectrum:
                axspec = fig.add_axes([0.10,0.15,0.5,0.75])
                axim   = fig.add_axes([0.65,0.15,0.26,0.75])
                axspec.set_xlabel(r"Wavelength", fontsize="large")
                axspec.set_ylabel(r"Flux", fontsize="large")
            else:
                axim   = fig.add_axes([0.12,0.12,0.8,0.8])
                
        elif show_meanspectrum:
            axspec, axim = ax
            fig = axspec.figure
        else:
            axim = ax
            fig  = axim.figure

        # - Ploting
        self._display_im_(axim, toshow, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        self._display_spec_(axspec, toshow,  **kwargs)
        if show_meanspectrum:
            axspec.text(0.03,0.95, "Mean Spectrum", transform=axspec.transAxes,
                            va="top", ha="left",
                          bbox={"facecolor":"w", "alpha":0.5,"edgecolor":"None"})
        
        # - out
        fig.figout(savefile=savefile, show=show)

    # --------- #
    # Internal  #
    # --------- #
    def _display_im_(self, axim, toshow="data", cmap=None,
                    lbdalim=None, interactive=False,
                     vmin=None, vmax=None, **kwargs):
        """ """
        if axim is None:
            return
        
        if not interactive and self.is_3d_cube():
            # - let's do the simple thing for this usual case
            return axim.imshow(np.sum(eval("self.%s"%toshow), axis=0), origin="lower",
                        interpolation="nearest",cmap=cmap,aspect='auto', **kwargs)
        
        # ----------------
        # Internal Imshow
        from matplotlib import patches
        # - which colors
        colors = self._data_to_color_(toshow, cmap=cmap, lbdalim=lbdalim,
                                      vmin = vmin,
                                      vmax = vmax)
        # - The Patchs
        ps = [patches.Polygon( self.spaxel_vertices+np.asarray(self.index_to_xy(id_)),
                        facecolor=colors[i], alpha=0.8,**kwargs) for i,id_  in enumerate(self.indexes)]
        ip = [axim.add_patch(p_) for p_ in ps]
        axim.autoscale(True, tight=True)
        return ip

    def _data_to_color_(self, toshow="data", lbdalim=None,
                        cmap=None, vmin=None, vmax=None, **kwargs):
        """ Convert the given data into colors.
        This will convert the `data` -> [0,1] scale and then
        feed that to the matplotlib colormap.

        Parameters
        ----------
        toshow: [string] -optional-
            which property will be used to define the color.
            
        lbdalim: [None/2D array] -optional-
            Lbda limit to be used. 
            - None: No limit
            - [min/None ; max/None] lower or upper limit (or None if None)
        
        vmin, vmax: [float /string / None]
            Upper and lower value for the colormap. 3 Format are available
            - float: Value in data unit
            - string: percentile. Give a float (between 0 and 100) in string format.
                      This will be converted in float and passed to numpy.percentile
            - None: The default will be used (percentile 0.5 and 99.5 percent respectively).
            (NB: vmin and vmax are independent, i.e. one can be None and the other '98' for instance)
            
        cmap: [matplotlib colormap] -optional-
            If None, viridis will be used.
        
        **kwargs  goes to get_slice(): e.g. usemean
        Returns
        -------
        RGBA array (len of `data`)
        """
        # - value limits 
        if lbdalim is None:
            lbdalim = [None,None]
        elif np.shape(lbdalim) != (2,):
            raise TypeError("lbdalim must be None or [min/max]")
        
        # - Scaling used
        mean_flux = self.get_slice(lbda_min=lbdalim[0], lbda_max=lbdalim[1], data=toshow, **kwargs)
        
        # - vmin / vmax trick
        vmin = np.nanpercentile(mean_flux, 0.5) if vmin is None else \
          np.nanpercentile(mean_flux, float(vmin)) if type(vmin) == str else\
          vmin
        vmax = np.nanpercentile(mean_flux, 99.5) if vmax is None else \
          np.nanpercentile(mean_flux, float(vmax)) if type(vmax) == str else\
          vmax
          
        # - colormap used
        if cmap is None:
            from matplotlib.pyplot import cm
            cmap = cm.viridis
            
        return cmap( (mean_flux-vmin)/(vmax-vmin) )
        
    def _display_spec_(self, axspec, toshow="data", **kwargs):
        """ """
        if axspec is not None:
            if self.is_3d_cube():
                spec = np.nanmean(np.concatenate(eval("self.%s"%toshow).T), axis=0)
                var  = np.nanmean(np.concatenate(self.variance.T), axis=0) / np.prod(self.data.shape[1:]) \
                  if 'data' in toshow and self.has_variance() else None
            else:
                spec = np.nanmean(eval("self.%s"%toshow).T, axis=0)
                var  = np.nanmean(self.variance.T, axis=0) / np.prod(self.data.shape[1:]) \
                  if 'data' in toshow and self.has_variance() else None
            axspec.specplot(self.lbda, spec, var=var)
            
        
    # =================== #
    #   Properties        #
    # =================== #
    # -----------
    # Side Prop
    @property
    def adr(self):
        """ If loaded (see load_adr) if object contains the methods to get the evolution 
        of a sky position as a function of wavelength accounting for 
        atmospherical differential refraction (ADR) """
        return self._side_properties["adr"]
    
    def load_adr(self, airmass,  parangle, temperature, relathumidity,
                     pressure=630, lbdaref=5000):
        """ 
        airmass: [float]
            Airmass of the target

        parangle: [float]
            Parralactic angle in degree

        temperature: [float]
            temperature in Celcius
            
        relathumidity: [float <100]
            Relative Humidity in %
            
        pressure: [float] -optional- 
            Air pressure in mbar
            
        lbdaref: [float] -optional-
            Reference wavelength, the position shift will be given as a function 
            of this wavelength.
        
        """
        from .adr import get_adr
        self._side_properties["adr"] = \
          get_adr(airmass=airmass, parangle=parangle,
                      temperature=temperature, relathumidity=relathumidity,
                     pressure=pressure, lbdaref=lbdaref)
    # -----------
    # - internal
    @property
    def _w_spix(self):
        if "wspix" not in self.spec_prop or self.spec_prop["wspix"] is None:
            self.spec_prop["wspix"] = np.shape(self.data)[1]
        return self.spec_prop["wspix"]

    @property
    def _n_spix(self):
        if "nspix" not in self.spec_prop or self.spec_prop["nspix"] is None:
            self.spec_prop["nspix"] = np.shape(self.data)[2]
        return self.spec_prop["nspix"]
