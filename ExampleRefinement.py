import cubit
cubit.init(['cubit','-nojournal'])
cubit.reset()

cubit.cmd('cd \'C:\\Users\\vmh5\\Documents\\CubitPython4SPEED\'')
#configuration file f with the option curve_refinement=True :
f="Files\TopoExampleRefinement.cfg"
# file is the merged unrefined mesh:
file="C:/Users/vmh5/Documents/CubitPython4SPEED/Example outputs/TopoExample_vol_merged.cub5"
#here you load the mesh in Cubit:
cubit.cmd('import cubit "' + file + '"')
cubit.cmd('set echo off')
cubit.cmd('set info off')
import main
#run the refinement routine:
main.basin_refinement(f)
