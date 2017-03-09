#! /usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import numpy as np
from scipy.spatial import distance
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as mpl

from propobject import BaseObject

from .spectroscopy import Cube
from .tools        import kwargs_update



DOCUMENTATION = \
   """
   # Concept
   
   Navigate through an IFU cube using spectrum and stacked-image panels.

   # Select Spaxels
   
   The spaxels you select on the 'imshow' like axis will be used 
   to display the corresponding spectrum
   - mouse-clic: to select 1 spaxel
   - mouse-drag: draw the diagonal of a rectangle, select the spaxels within it
   - `shift`  + mouse-drag: draw a polygon with the mouse to select the spaxels within it
   - `control`+ mouse-drag: the distance between the picked and release points 
                            to define a Circle. (center = picked position).
                            Spaxels within this circle will be selected

   => hold the "super" keyboard key avoid clearing the axis while 
      plotting a newly picked spaxel

   # Select Wavelengthes
   
   Select a wavelength range on the 'spectrum-axis' to affect the color mapping
   on the 'imshow-like axis'. 
   - drag-mouse: select a wavelength range
   - single-clic: remove the wavelength selection.

   # Keyboad keys
   
   - escape: reset the figue 
   - shift: host this key to allow alternative mouse actions
   - h: print this help
   - super: hold this key to display a new picked spaxels without clearing the axis.

   # Credits
   
   pyifu ; Mickael Rigault
   """



WAVESPAN_PROP = dict(edgecolor=mpl.cm.binary(0.7,0.7), facecolor=mpl.cm.binary(0.7,0.2))

class InteractiveCube( BaseObject ):
    """ """
    PROPERTIES = ["cube","figure","axspec","axim"]

    DERIVED_PROPERTIES = ["property_backup"]
    
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
            figsizes = self.fig.get_size_inches()
            axspec = self.fig.add_axes([0.10,0.15,0.5,0.75])
            axim   = self.fig.add_axes([0.65,0.15,0.75*figsizes[1]/float(figsizes[0]),0.75])
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
        from .tools import ipython_info
        if not self.has_figure():
            raise ValueError("define a figure first")
        # -- let's go -- #
        self._spaxels = self.cube._display_im_(self.axim, interactive=True, linewidth=0, edgecolor="0.7")
        
        self._current_spectra = []
        self._current_spectra.append(self.cube._display_spec_(self.axspec))
        
        self.setup()

        
        self.interact_connect(axes_enter_event     = self.interact_enter_axis,
                              axes_leave_event     = self.interact_leave_axis,
                              
                              button_press_event   = self.interact_onclick,
                              motion_notify_event  = self.interact_trackmotion,
                              button_release_event = self.interact_onrelease,
                              
                              key_press_event      = self.interact_presskey,
                              key_release_event    = self.inteact_releasekey)
        
        if ipython_info() == "notebook":
            self.fig.show()
        else:
            mpl.show()
        
    def interact_connect(self, **kwargs):
        """ More method that connect the event and the actions """
        [self.fig.canvas.mpl_connect(key,val) for key,val in kwargs.iteritems()]


    def set_to_origin(self):
        """ Set the internal parameters to their initial values. """
        #   PICKING    #
        self._clicked         = False
        self._picked_poly     = None
        self._currentactive   = 0
        self.selected_spaxels = []
        #   KEYBOARD   #
        self.pressed_key      = {}
        #   AXIM       #
        #   AXSPEC     #
        self._current_spectra     = []
        self._selected_wavelength = None
        self._axspec_span         = []
        
    def reset(self):
        """ Clean the Axes and set things back to there initial values """
        self.change_axim_color(None) # Remove selected wavelength
        self._clean_picked_im_() # Remove Spaxels selected
        self.clean_axspec(draw=False) # Clear the Axis
        self.cube._display_spec_(self.axspec) # Draw new spectra
        self.axspec.set_ylim(self.get_autospec_ylim()) # good ylim
        self.set_to_origin() # Set back the class to initial conditions
        self.fig.canvas.draw()
        
    def setup(self):
        """ Run That when you launch. """

        self.set_to_origin()
        
        # - Default Fixed Values
        self.property_backup[self.axspec]["default_axedgewidth"] = \
          [s_.get_linewidth() for s_ in self.axspec.spines.values()]

        self.property_backup[self.axim]["default_axedgewidth"] = \
          [s_.get_linewidth() for s_ in self.axim.spines.values()]
          
        self.property_backup[self.axim]["default_edgecolor"] = \
              self._spaxels[0].get_edgecolor()
        self.property_backup[self.axim]["default_zordercolor"] = \
              self._spaxels[0].get_zorder()

        self.property_backup[self.axim]["default_facecolors"] = \
              [sp.get_facecolor() for sp in self._spaxels]
              
        self._default_z_order_spaxels   = self._spaxels[0].get_zorder()
        self._default_linewidth_spaxels = self._spaxels[0].get_linewidth()
        
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
        if self.fig.canvas.manager.toolbar._active is not None:
            return
        
        if event.inaxes == self.axim:
            self._onclick_axim_(event)
        elif event.inaxes == self.axspec:
            self._onclick_axspec_(event)

    def interact_onrelease(self, event):
        """ What would happen when you release the mouse click """
        if self.fig.canvas.manager.toolbar._active is not None:
            return
        
        if event.inaxes == self.axim:
            self._onrelease_axim_(event)
        elif event.inaxes == self.axspec:
            self._onrelease_axspec_(event)

    def interact_trackmotion(self, event):
        if self.fig.canvas.manager.toolbar._active is not None:
            return
        
        if event.inaxes == self.axim:
            self._ontrack_axim_(event)
            
    # - axim
    def _onclick_axim_(self, event):
        """ """
        self._clicked = True
        self._picked_spaxel   = [event.xdata, event.ydata]
        self._tracked_spaxel  = []
        
    def _ontrack_axim_(self, event):
        """ What happen with the mouse goes over the axis?"""
        if self._clicked:
            # Here, minimal parameter needed to set the polygon are defined.
            # The actual polygon is defined during the key release.
            
            # Any Polygone
            if self._keyshift:
                self._tracked_spaxel.append([event.xdata, event.ydata])
                self._picked_poly = ["Polygon",   [self._tracked_spaxel]]
                
            # A Circle
            elif self._keycontrol:
                self._picked_poly = ["Circle",  [self._picked_spaxel,distance.euclidean(self._picked_spaxel, [event.xdata, event.ydata])]]
            # Rectangle
            else:
                self._picked_poly = ["Polygon", [[[self._picked_spaxel[0], self._picked_spaxel[1]],
                                                  [self._picked_spaxel[0],event.ydata],
                                                    [event.xdata,event.ydata], [event.xdata,self._picked_spaxel[1]]]]]
            
    def _onrelease_axim_(self, event):
        """ """
        # - Region Selection
        if self._picked_poly is not None:
            which, args = self._picked_poly
            poly = eval("patches.%s(*args)"%(which))
            self.selected_spaxels  = [i for i,id_ in enumerate(self.cube.indexes)
                                        if poly.contains_point( self.cube.index_to_xy(id_)) ]
            self._picked_poly = None
            
        # - Simple Picking
        else:
            self.selected_spaxels  = [np.argmin([distance.euclidean(self.cube.index_to_xy(i),[event.xdata, event.ydata])
                                                    for i in self.cube.indexes])]
            
        # - What to do with the selected spaxels            
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
        self._selected_wavelength = [event.xdata]
            
    def _onrelease_axspec_(self, event):
        """ """
        if self._selected_wavelength is None:
            return
        
        if self._selected_wavelength[0] == event.xdata:
           self._clean_wave_axspec_()
           self._selected_wavelength = None
        else:
            self._selected_wavelength.append(event.xdata)
            self.draw_selected_wavelength()

        self.change_axim_color(self._selected_wavelength)
        self.fig.canvas.draw()
        
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
        else:
            self._currentactive +=1 
        # - Show me the selected spaxels            
        for p in self.selected_spaxels:
            self._spaxels[p].set_zorder(self._default_z_order_spaxels+1)
            self._spaxels[p].set_linewidth(self._default_linewidth_spaxels+1)
            self._spaxels[p].set_edgecolor(self._active_color)
            
        self._holded_spaxels.append(self.selected_spaxels)
        
    def _clean_picked_im_(self):
        """ Removes the changes that have been made """
        if not hasattr(self, "_holded_spaxels"):
            self._holded_spaxels = []
            return
        self._currentactive = 0
        for older_spaxels in self._holded_spaxels:
            for p in older_spaxels:
                self._spaxels[p].set_zorder(self._default_z_order_spaxels)
                self._spaxels[p].set_linewidth(self._default_linewidth_spaxels)
                self._spaxels[p].set_edgecolor(self.property_backup[self.axim]["default_edgecolor"])
                
        self._holded_spaxels = []
        
    # -------- #
    #  axspec  #
    # -------- #
    def show_picked_spectrum(self):
        """ """
        if not self._hold:
            self._clean_spec_axspec_()
        spec = self.cube.get_spectrum(self.selected_spaxels)
        self._current_spectra.append(spec.show(ax=self.axspec, color=self._active_color, show=False))
        self.axspec.set_ylim(self.get_autospec_ylim())

    def get_autospec_ylim(self):
        """ """
        edge = 0.1
        if len(self.axspec.get_lines())>0:
            ymin, ymax = np.percentile(np.concatenate([l.get_data()[-1] for l in self.axspec.get_lines()]), [0,100])
            ymin =ymin*(1.+edge) if ymin<0 else ymin*(1-edge)
            ymax =ymax*(1.+edge) if ymax>0 else ymax*(1-edge)
            
            return ymin,ymax
        return -1,1
    
    def draw_selected_wavelength(self):
        """ """
        self._clean_wave_axspec_()
        self._axspec_span.append(self.axspec.axvspan(self._selected_wavelength[0],self._selected_wavelength[1],
                                                         **WAVESPAN_PROP))

    def change_axim_color(self, lbdalim):
        """ """
        colors = self.cube._data_to_color_("data", lbdalim=lbdalim)
        [s.set_facecolor(c) for s,c in zip(self._spaxels, colors)]
        
        
    def clean_axspec(self, draw=False):
        self._clean_spec_axspec_()
        self._clean_wave_axspec_()
        if draw:
            self.fig.canvas.draw()
        
    def _clean_spec_axspec_(self):
        """ """
        self.axspec.lines = [] # remove the spectra lines
        self.axspec.collections = [] # and their variance
        
    def _clean_wave_axspec_(self):
        """ Remove the selected wavelength """
        if self._axspec_span is not None:
            for axs in self._axspec_span:
                try: # Sometime its fails...
                    axs.remove()
                except:
                    pass
        
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

    # - On Flight Properties
    @property
    def _keyshift(self):
        """ If the shift key pressed? """
        return self.pressed_key.get("shift",False)
    
    @property
    def _keycontrol(self):
        """ If the control key pressed? """
        return self.pressed_key.get("control",False)
    
    @property
    def _active_color(self):
        c = self._currentactive%10 
        return "k" if c == 0 else "C%d"%c
    
    @property
    def _hold(self):
        """ """
        return self.pressed_key.get("super",False)




