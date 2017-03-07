#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Internal small toolbox"""

import numpy as np
import matplotlib.pyplot as mpl

__all__ = ["kwargs_update","figout","specplot"]

def kwargs_update(default,**kwargs):
    """
    """
    k = default.copy()
    for key,val in kwargs.iteritems():
        k[key] = val
        
    return k

def make_method(obj):
    """Decorator to make the function a method of *obj*.

    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate

def ipython_info():
    import sys
    return 'notebook' if 'ipykernel' in sys.modules \
      else "terminal" if 'Ipython' in sys.modules \
    else None
    
# ================== #
#   MPL Add On       #
# ================== #

    






    
@make_method(mpl.Axes)
def specplot(ax,x,y,var=None,
             color=None,bandprop={},
             err_onzero=False,**kwargs):
    """This function in a build-in axes method that enable to quickly and
    easily plot a spectrum.
    """
    # -----------------------
    # - Properties of plot
    default_kwargs = dict(
        color=mpl.cm.Blues(0.8),
        ls="-",lw=1,marker=None,zorder=6,
        )
    if color is not None:
        default_kwargs["color"] = color
    propplot = kwargs_update(default_kwargs,**kwargs)
    # -- Plot 
    pl = ax.plot(x,y,**propplot)
    
    # -----------------------
    # - Properties of band
    if var is not None:
        default_band   = dict(
            color=propplot["color"],alpha=0.3,
            zorder=3,label="_no_legend_"
            )
        bandprop = kwargs_update(default_band,**bandprop)
        # -- Band
        if not err_onzero:
            fill = ax.fill_between(x,y+np.sqrt(var),y-np.sqrt(var),
                            **bandprop)
        else:
            fill = ax.fill_between(x,np.sqrt(var),-np.sqrt(var),
                            **bandprop)
    else:
        fill = None
        
    return pl,fill

@make_method(mpl.Figure)
def figout(fig,savefile=None,show=True,add_thumbnails=False,
           dpi=200):
    """This methods parse the show/savefile to know if the figure
    shall the shown or saved."""
    
    if savefile in ["dont_show","_dont_show_","_do_not_show_"]:
        show = False
        savefile = None

    if savefile is not None:
        if not savefile.endswith(".pdf"):
            extention = ".png" if not savefile.endswith(".png") else ""
            fig.savefig(savefile+extention,dpi=dpi)
            
        if not savefile.endswith(".png"):
            extention = ".pdf" if not savefile.endswith(".pdf") else ""
            fig.savefig(savefile+extention)
            
        if add_thumbnails:
            fig.savefig(savefile+"_thumb"+'.png',dpi=dpi/10.)
            
    elif show:
        fig.canvas.draw()
        fig.show()
