#! /usr/bin/env python
# -*- coding: utf-8 -*-


#################################
#
#   MAIN 
#
#################################
if  __name__ == "__main__":
    
    import argparse
    from pyifu import load_cube
    # ================= #
    #   Options         #
    # ================= #
    parser = argparse.ArgumentParser(
        description=""" run the interactive plotting of a given cube""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('infile', type=str, default=None,
                        help='cube filepath')
    
    parser.add_argument('--rmsky',  action="store_true", default=False,
                        help='Removes the sky component from the cube')
    
    parser.add_argument('--nskyspaxels',  type=int, default=None,
                        help='Number of faintest spaxels used to estimate the sky. Default 10% of spaxels')

    # // Ploting
    parser.add_argument('--vmin',  type=str, default="2",
                        help='Data Percentage used for imshow "vmin" when using the --display mode')
    
    parser.add_argument('--vmax',  type=str, default="98",
                        help='Data Percentage used for imshow "vmax" when using the --display mode')

    args = parser.parse_args()
    # ================= #
    #  The Scripts      #
    # ================= #
    cube = load_cube(args.infile)
    
    if args.rmsky:        
        nspaxels = int(len(cube.nspaxels)/10) if args.nskyspaxels or args.nskyspaxels in ["None"] else  args.nskyspaxels

        cube._sky = cube.get_spectrum(cube.get_faintest_spaxels(nspaxels),
                                          usemean=False)
        cube.remove_flux( cube._sky.data)

    cube.show(interactive=True, notebook=False, vmin=args.vmin, vmax=args.vmax)
