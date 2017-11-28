#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy         as np

from propobject import BaseObject

""" Atmospheric Differential Refraction: Evolution of the spatial position as a function of wavelength. 

Credit: Y.Copin (y.copin@ipnl.in2p3.fr)
Adapted by: M. Rigault

Inspiration: SNfactory ToolBox (see also Don Neill's Fork of KPY)
Original Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp

"""

__all__ = ["get_adr"]

def get_adr(**kwargs):
    """ returns an instance of a ARD object.

    **kwargs: 
    Set any of the fundamental properties:
        - airmass []
        - lbdaref [A]
        - pressure [mbar]
        - parangle [deg]
        - relathumidity [%]
        - temperature [C]

        for instance to set the airmarss to 1.4, simply do:
        self.set(airmass=1.4).
        
    Returns
    -------
    ADR
    """
    return ADR(**kwargs)

    
class ADR( BaseObject ):
    """ """
    PROPERTIES = ["airmass",  "lbdaref","parangle", "pressure",
                 "relathumidity" , "temperature"]
    DERIVED_PROPERTIES = []

    def __init__(self, **kwargs):
        """  **kwargs: 
        Set any of the fundamental properties:
        - airmass []
        - lbdaref [A]
        - pressure [mbar]
        - parangle [deg]
        - relathumidity [%]
        - temperature [C]

        for instance to set the airmarss to 1.4, simply do:
        self.set(airmass=1.4).
        """
        if kwargs:
            self.set(**kwargs)
    
    # =================== #
    #   Methods           #
    # =================== #
    def refract(self, x, y, lbda, backward=False, unit=1.,
                    **kwargs):
        """If forward (default), return refracted position(s) at
        wavelength(s) *lbda* [Å] from reference position(s) *x*,*y*
        (in units of *unit* in arcsec).  Return shape is
        (2,[nlbda],[npos]), where nlbda and npos are the number of
        input wavelengths and reference positions.

        If backward, one should have `len(x) == len(y) ==
        len(lbda)`. Return shape is then (2,npos).

        Coordinate *x* is counted westward, and coordinate *y* counted
        northward (standard North-up, Est-left orientation).

        Anonymous `kwargs` will be propagated to :meth:`set`.
        """
        if kwargs:                      # Update parameters if needed
            self.set(**kwargs)

        x0 = np.atleast_1d(x)                 # (npos,)
        y0 = np.atleast_1d(y)
        if np.shape(x0) != np.shape(y0):
            raise TypeError("x and y do not have the same shape")
        
        npos = len(x0)
        
        dz = self._delta * self.get_scale(lbda) / unit  # [unit]

        if backward:
            nlbda = len(np.atleast_1d(lbda))
            assert npos == nlbda, "Incompatible x,y and lbda vectors."
            x = x0 - dz * np.sin(self._parangle_rad)
            y = y0 + dz * np.cos(self._parangle_rad)  # (nlbda=npos,)
            out = np.vstack((x, y))           # (2,npos)
        else:
            dz = dz[:, np.newaxis]            # (nlbda,1)
            x = x0 + dz * np.sin(self._parangle_rad)  # (nlbda,npos)
            y = y0 - dz * np.cos(self._parangle_rad)  # (nlbda,npos)
            out = np.dstack((x.T, y.T)).T     # (2,nlbda,npos)

        return out.squeeze()                 # (2,[nlbda],[npos])

        
    # ------- #
    # SETTER  #
    # ------- #
    def set(self, **kwargs):
        """ 
        Set any of the fundamental properties:
        - airmass []
        - lbdaref [A]
        - pressure [mbar]
        - parangle [deg]
        - relathumidity [%]
        - temperature [C]

        for instance to set the airmarss to 1.4, simply do:
        self.set(airmass=1.4).

        _Important_: only the aforementioned fundametal properties could be set.
        
        """
        for k,v in kwargs.items():
            if k not in self.PROPERTIES:
                raise ValueError("unknown property %s, it cannot be set. known properties are: ",", ".join(self.PROPERTIES))
            self._properties[k] = v


    # ------- #
    # GETTER  #
    # ------- #
    def get_scale(self, lbda, **kwargs):
        """
        Return ADR scale [arcsec] for wavelength(s) `lbda` [A].

        Anonymous `kwargs` will be propagated to :meth:`set`.
        """
        if kwargs:                      # Update parameters if needed
            self.set(**kwargs)
            
        lbda = np.atleast_1d(lbda)       # (nlbda,)

        # Approximate ADR to 1st order in (n - 1). The difference
        # between exact expression and 1st-order approximation reaches
        # 4e-9 at 3200 A.
        # dz = self.nref - \
        #      refractiveIndex(lbda, P=self.P, T=self.T, RH=self.RH)
        
        # Exact ADR expression
        dz = (self.get_refractive_index(lbda)**-2 - self.nref**-2) * 0.5

        return dz * 180. * 3600 / np.pi    # (nlbda,) [arcsec]

    def get_refractive_index(self, lbda, **kwargs ):
        """ relative index for the given wavelength. 

        Anonymous `kwargs` will be propagated to :meth:`set`.
        """
        if kwargs:                      # Update parameters if needed
            self.set(**kwargs)
            
        return refractive_index(lbda,
                                pressure=self.pressure, temperature=self.temperature,
                                relathumidity=self.relathumidity)
    

        
    # =================== #
    #   Properties        #
    # =================== #
    @property
    def temperature(self):
        """ temperature in Celcius """
        return self._properties["temperature"]

    @property
    def parangle(self):
        """ parralactic angle in degree """
        return self._properties["parangle"]
    
    @property
    def pressure(self):
        """ Pressure in mm Hg """
        return self._properties["pressure"]

    @property
    def airmass(self):
        """ Airmass of the target """
        return self._properties["airmass"]
    
    @property
    def relathumidity(self):
        """ Relative Humidity """
        return self._properties["relathumidity"]

    @property
    def lbdaref(self):
        """ Relative Humidity """
        return self._properties["lbdaref"]

    # --------- #
    #  derived  #
    # --------- #
    @property
    def nref(self):
        """ reference wavelength refractive index """
        return self.get_refractive_index(self.lbdaref)
    
    @property
    def _delta(self):
        """ For the refract method: np.tan(np.arccos(1. / self.airmass)) """
        return np.tan(np.arccos(1. / self.airmass))
    
    @property
    def _parangle_rad(self):
        """ parralactic angle in radian """
        return self.parangle / 180. * np.pi
##############################
#                            #
#   General Functions        #
#                            #
##############################

def refractive_index(lbda, pressure=617., temperature=2., relathumidity=0):
    """Compute refractive index at vacuum wavelength.
    source: NIST/IAPWS via SNfactory ToolBox
            http://emtoolbox.nist.gov/Wavelength/Documentation.asp

    Parameters:
    -----------
    lbda: [float / array of]
        wavelength in [Å]
        
    pressure: [float] -optional-
        air pressure in mbar

    temperature: [float] -optional-
        air temperature in Celcius

    relathumidity: [float] -optional-
        air relative humidity in percent.
        [the water vapor pressure will be derived from it,
        according to (modified) Edlén Calculation of the Index of
        Refraction from NIST 'Refractive Index of Air Calculator'.  CO2
        concentration is fixed to 450 µmol/mol.]

    Returns
    -------
    float/array (depending on lbda input)
    """

    A = 8342.54
    B = 2406147.
    C = 15998.
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661

    iml2 = (lbda * 1e-4)**-2              # 1/(lambda in microns)**2
    nsm1e2 = 1e-6 * (A + B / (130. - iml2) + C / (38.9 - iml2))  # (ns - 1)*1e2
    # P in mbar = 1e2 Pa
    X = (1. + 1e-6 * (E - F * temperature) * pressure) / (1. + G * temperature)
    n = 1. + pressure * nsm1e2 * X / D             # ref. index corrected for P,T

    if relathumidity:                              # Humidity correction
        pv = relathumidity / 100. * saturation_vapor_pressure(temperature)  # [Pa]
        n -= 1e-10 * (292.75 / (temperature + 273.15)) * (3.7345 - 0.0401 * iml2) * pv

    return n


def saturation_vapor_pressure(temperature):
    """Compute saturation vapor pressure [Pa] for temperature *T* [°C]
    according to Edlén Calculation of the Index of Refraction from
    NIST 'Refractive Index of Air Calculator'.

    Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """

    t = np.atleast_1d(temperature)
    psv = np.where(t >= 0,
                  _saturationVaporPressureOverWater(temperature),
                  _saturationVaporPressureOverIce(temperature))

    return psv       # [Pa]    


def _saturationVaporPressureOverIce(temperature):
    """See :func:`saturation_vapor_pressure`"""

    A1 = -13.928169
    A2 = 34.7078238
    th = (temperature + 273.15) / 273.16
    Y = A1 * (1 - th**-1.5) + A2 * (1 - th**-1.25)
    psv = 611.657 * np.exp(Y)

    return psv

def _saturationVaporPressureOverWater(temperature):
    """See :func:`saturation_vapor_pressure`"""

    K1 = 1.16705214528e+03
    K2 = -7.24213167032e+05
    K3 = -1.70738469401e+01
    K4 = 1.20208247025e+04
    K5 = -3.23255503223e+06
    K6 = 1.49151086135e+01
    K7 = -4.82326573616e+03
    K8 = 4.05113405421e+05
    K9 = -2.38555575678e-01
    K10 = 6.50175348448e+02

    t = temperature + 273.15                      # °C → K
    x = t + K9 / (t - K10)
    A = x**2 + K1 * x + K2
    B = K3 * x**2 + K4 * x + K5
    C = K6 * x**2 + K7 * x + K8
    X = -B + np.sqrt(B**2 - 4 * A * C)
    psv = 1e6 * (2 * C / X)**4

    return psv










    

def air_index(lam, pressure=600, temperature=7,
                  f=8):
    """ Returns index of refraction of air-1 at
        lam in micron at vacuum
        p is pressure in mm Hg
        t is temperature in deg C
        f is water vapor pressure in mm Hg
    """

    k1 = (1/lam)**2
    nm1e6 = 64.328 + 29498.1/(146-k1) + 255.4/(41-k1)

    nm1e6 *= pressure * (1 + (1.049-0.0157 * temperature)*1e-6*pressure) / (720.883 * (1 + 0.003661 * temperature))

    nm1e6 -= 0.0624 - 0.000680 * k1 / (1 + 0.003661 * temperature) * f

    return nm1e6/1e6


def atm_disper(l2, l1, airmass, **kwargs):
    """ atmospheric dispersion in arcsecond between l2 and l1 in micron
        at a given airmass. See air index for documentation on pressure,
        temperature, and water vapor pressure"""

    z = np.arccos(1.0/airmass)
    return 206265 * (air_index(l2, **kwargs) - air_index(l1,
                                                         **kwargs)) * np.tan(z)






def air_index(lam, pressure=600, temperature=7,
                  f=8):
    """ Returns index of refraction of air-1 at
        lam in micron at vacuum
        p is pressure in mm Hg
        t is temperature in deg C
        f is water vapor pressure in mm Hg
    """

    k1 = (1/lam)**2
    nm1e6 = 64.328 + 29498.1/(146-k1) + 255.4/(41-k1)

    nm1e6 *= pressure * (1 + (1.049-0.0157 * temperature)*1e-6*pressure) / (720.883 * (1 + 0.003661 * t))

    nm1e6 -= 0.0624 - 0.000680 * k1 / (1 + 0.003661 * temperature) * f

    return nm1e6/1e6


def atm_disper(l2, l1, airmass, **kwargs):
    """ atmospheric dispersion in arcsecond between l2 and l1 in micron
        at a given airmass. See air index for documentation on pressure,
        temperature, and water vapor pressure"""

    z = np.arccos(1.0/airmass)
    return 206265 * (air_index(l2, **kwargs) - air_index(l1,
                                                         **kwargs)) * np.tan(z)
