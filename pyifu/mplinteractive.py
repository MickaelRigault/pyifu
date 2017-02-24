#! /usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import numpy as np

import matplotlib
from matplotlib import patches
import matplotlib.pyplot as mpl

from propobject import BaseObject

from .spectroscopy import Cube
from .tools        import kwargs_update



DOCUMENTATION = \
   """
   # Concept
   Show the IFU cube using spectrum- and stacked-image panels

   # Select Spaxels
   The spaxels you select on the 'imshow' like axis will be used 
   to display the corresponding spectrum

   - mouse-clic: to select 1 spaxel
   - mouse-drag: draw the diagonal of a rectangle, select the spaxels within it
   - `shift` + mouse-drag: draw a polygon with the mouse to select the spaxels within it

   # Keyboad keys
   - escape: reset the figue 
   - shift: host this key to allow alternative mouse actions
   - h: print this help

   # Credits
   pyifu ;  Mickael Rigault
   """



class InteractiveCube( BaseObject ):
    """ """
    PROPERTIES = ["cube",
                  "figure","axspec","axim"]

    DERIVED_PROPERTIES = ["property_backup","hold"]
    
    def __init__(self, cube, fig=None, axes=None):
        """ """
        if cube is not None:
            self.set_cube(cube)
        if axes is not None:
            self.set_axes(axes)
        else:
            self.set_figure(fig)
            self.set_axes(axes)
            
    # =================== #
    #   Main Methods      #
    # =================== #
    # ---------- #
    #  SETTER    #
    # ---------- #
    def set_cube(self, newcube):
        """ Attach a pyifu Cube to this instance """
        self._properties["cube"] = newcube
    
    def set_figure(self, fig=None, **kwargs):
        """ attach a figure to this method. """
        if fig is None:
            figprop = kwargs_update(dict(figsize=[10,3.5]), **kwargs)
            self._properties["figure"] = mpl.figure(**figprop)
        elif matplotlib.figure.Figure not in fig.__class__.__mro__:
            raise TypeError("The given fig must be a matplotlib.figure.Figure (or child of)")
        else:
            self._properties["figure"] = mpl.figure(**figprop)

    def set_axes(self, axes=None, **kwargs):
        """ """
        if axes is None and not self.has_figure():
            raise AttributeError("If no axes given, please first set the figure.")
        
        if axes is None:
            axspec = self.fig.add_axes([0.10,0.15,0.5,0.75])
            axim   = self.fig.add_axes([0.65,0.15,0.26,0.75])
            axspec.set_xlabel(r"Wavelength", fontsize="large")
            axspec.set_ylabel(r"Flux", fontsize="large")
            self.set_axes([axspec,axim])
            
        elif len(axes) != 2:
            raise TypeError("you must provide 2 axes [axspec and axim] and both have to be matplotlib.axes(._axes).Axes (or child of)")
        else:
            # - actual setting
            self._properties["axspec"], self._properties["axim"] = axes
            if not self.has_figure():
                self.set_figure(self.axspec.figure)
            

    # ============================== #
    #  Low Level Connection Magic    #
    # ============================== #
    def launch(self):
        """ """
        if not self.has_figure():
            raise ValueError("define a figure first")
        # -- let's go -- #
        self._spaxels = self.cube._display_im_(self.axim, interactive=True)
        self.cube._display_spec_(self.axspec)

        self.setup()
        
        self.interact_connect(axes_enter_event     = self.interact_enter_axis,
                              axes_leave_event     = self.interact_leave_axis,
                              
                              button_press_event   = self.interact_onclick,
                              motion_notify_event  = self.interact_trackmotion,
                              button_release_event = self.interact_onrelease,
                              
                              key_press_event      = self.interact_presskey,
                              key_release_event    = self.inteact_releasekey)
        self.fig.show()
        
    def interact_connect(self, **kwargs):
        """ More method that connect the event and the actions """
        [self.fig.canvas.mpl_connect(key,val) for key,val in kwargs.iteritems()]


    def reset(self):
        """ """
        self._clean_picked_im_()
        self.axspec.cla()
        self.cube._display_spec_(self.axspec)
        
        self._clicked         = False
        self.polyselected     = []
        self.selected_spaxels = []
        
    def setup(self):
        """ """
        self._clicked         = False
        self.polyselected     = []
        self.selected_spaxels = []
        # ------------ #
        #  Key Setup   #
        # ------------ #
        self.pressed_key = {}
        # ------------ #
        #  Axes Setup  #
        # ------------ #
        # - Default Values
        self.property_backup[self.axspec]["default_axedgewidth"] = \
          [s_.get_linewidth() for s_ in self.axspec.spines.values()]

        self.property_backup[self.axim]["default_axedgewidth"] = \
          [s_.get_linewidth() for s_ in self.axim.spines.values()]
          
        self.property_backup[self.axim]["default_edgecolor"] = \
              self._spaxels[0].get_edgecolor()
        self.property_backup[self.axim]["default_zordercolor"] = \
              self._spaxels[0].get_zorder()
        self._default_z_order_spaxels = self._spaxels[0].get_zorder()

        
    # -------------- #
    #  Information   #
    # -------------- #
    # ----------
    # - I/O Axes
    def interact_enter_axis(self, event):
        """ """
        if event.inaxes in [self.axim,self.axspec]:
            # - change axes linewidth
            [s_.set_linewidth(self.property_backup[event.inaxes]["default_axedgewidth"][i]*2)
                 for i,s_ in enumerate(event.inaxes.spines.values())]
            event.canvas.draw()
            
    def interact_leave_axis(self, event):
        """ """
        if event.inaxes in [self.axim,self.axspec]:
            # - change axes linewidth            
            [s_.set_linewidth(self.property_backup[event.inaxes]["default_axedgewidth"][i])
                 for i,s_ in enumerate(event.inaxes.spines.values())]
            event.canvas.draw()


    # ----------
    # Keyboard
    def interact_presskey(self, event):
        """ """
        if event.key in ["escape"]:
            self.reset()
            event.canvas.draw()
            return
        if event.key in ["h","H"]:
            print(DOCUMENTATION)
            return
        
        self.pressed_key[event.key] = True
        
    def inteact_releasekey(self, event):
        """ """
        self.pressed_key[event.key] = False
        
    # ----------
    # - Click 
    def interact_onclick(self, event):
        """ """
        if event.inaxes == self.axim:
            self._onclick_axim_(event)
        elif event.inaxes == self.axspec:
            self._onclick_axspec_(event)

    def interact_onrelease(self, event):
        """ What would happen when you release the mouse click """
        if event.inaxes == self.axim:
            self._onrelease_axim_(event)
        elif event.inaxes == self.axspec:
            self._onrelease_axspec_(event)

    def interact_trackmotion(self, event):
        if event.inaxes == self.axim:
            self._ontrack_axim_(event)

            
    # - axim
    def _onclick_axim_(self, event):
        """ """
        self._clicked = True
        self._picked_spaxel   = [np.round(event.xdata), np.round(event.ydata)]
        self._tracked_spaxel  = []
        
    def _ontrack_axim_(self, event):
        if self._clicked and u"shift" in self.pressed_key and self.pressed_key[u"shift"]:
            self._tracked_spaxel.append([event.xdata, event.ydata])
        
    def _onrelease_axim_(self, event):
        """ """
        if len(self._tracked_spaxel) >0:
            poly = patches.Polygon(self._tracked_spaxel)
            self.selected_spaxels  = [i for i in range(self.cube.nspaxels) if poly.contains_point( self.cube.index_to_xy(i))]
        # - With Tracking
        else:
            self._released_spaxel = np.round(event.xdata), np.round(event.ydata)
            x_min,x_max = np.sort([self._picked_spaxel[0],self._released_spaxel[0]])
            y_min,y_max = np.sort([self._picked_spaxel[1],self._released_spaxel[1]])
            self.selected_spaxels = np.concatenate([[self.cube.xy_to_index([x_,y_])[0]
                                                     for x_ in np.arange(x_min,x_max+1)]
                                                     for y_ in np.arange(y_min,y_max+1)])
            
        # ==== Show the selected spaxels
        if len(self.selected_spaxels)>0:
            self.show_picked_spaxels()
            self.show_picked_spectrum()
            event.canvas.draw()
        
        # - End of releasing event
        self._clicked = False
        
        
    # - axspec
    def _onclick_axspec_(self, event):
        """ """
    def _onrelease_axspec_(self, event):
        """ """
        
    # =================== #
    #   Affect Plot       #
    # =================== #
    # -------- #
    #   axim   #
    # -------- #
    def show_picked_spaxels(self):
        """ """
        # - if not hold. Remove the former this
        if not self._hold:
            self._clean_picked_im_()

        # - Show me the selected spaxels            
        for p in self.selected_spaxels:
            self._spaxels[p].set_zorder(self._default_z_order_spaxels+1)
            self._spaxels[p].set_edgecolor("0.7")
            
        self._holded_spaxels.append(self.selected_spaxels)

    def _clean_picked_im_(self):
        """ Removes the changes that have been made """
        if not hasattr(self, "_holded_spaxels"):
            self._holded_spaxels = []
            return
        
        for older_spaxels in self._holded_spaxels:
            for p in older_spaxels:
                self._spaxels[p].set_zorder(self._default_z_order_spaxels)
                self._spaxels[p].set_edgecolor(self.property_backup[self.axim]["default_edgecolor"])
                
        self._holded_spaxels = []

    # -------- #
    #  Spect   #
    # -------- #
    def show_picked_spectrum(self):
        """ """
        if not self._hold:
            self.axspec.cla()
        spec = self.cube.get_spectrum(self.selected_spaxels)
        spec.show(ax=self.axspec)
        
    # =================== #
    #   Properties        #
    # =================== #
    @property
    def cube(self):
        """ This cube that has to be displayed """
        return self._properties["cube"]

    # -----------
    # - Artists
    @property
    def fig(self):
        """ Matplotlib figure holding the axes """
        return self._properties["figure"]

    def has_figure(self):
        """ Tests if the figure has been set """
        return self.fig is not None

    @property
    def axspec(self):
        """ axes containing the selected spectrum """
        return self._properties["axspec"]
    
    @property
    def axim(self):
        """ axes containing the projected cube"""
        return self._properties["axim"]
    
    # ----------------
    # - Event Actions
    @property
    def property_backup(self):
        """ """
        if self._derived_properties["property_backup"] is None:
            self._derived_properties["property_backup"] = {self.fig   :{},
                                                           self.axim  :{},
                                                           self.axspec:{}
                                                            }
        return self._derived_properties["property_backup"]

    def hold(self):
        self._derived_properties["hold"] = ~ self._hold
        
    @property
    def _hold(self):
        if self._derived_properties["hold"] is None:
            self._derived_properties["hold"] = np.False_
        return self._derived_properties["hold"]
            




