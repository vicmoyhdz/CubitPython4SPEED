import cubit
cubit.init(['cubit','-nojournal'])
cubit.reset()

cubit.cmd('cd \'C:\\Users\\vmh5\\Documents\\CubitPython\'')
#configuration file f with the option curve_refinement=True :
f="Files\Refinement.cfg"
# file is the merged unrefined mesh:
file="C:/vol_merged.cub5"
#here you load the mesh in Cubit:
cubit.cmd('import cubit "' + file + '"')
cubit.cmd('set echo off')
cubit.cmd('set info off')
import main
#run the refinement routine:
main.basin_refinement(f)
